# Interferometer likelihood campaign

**Status:** open
**Question:** Where do the interferometer likelihoods — MGE, and the Delaunay-1500 / rectangular meshes on the sparse (W~) path — spend their time on the A100 and on CPU, and which levers pay?
**Pre-registered rule:** not recorded as a single campaign rule; each breakdown ranks levers from committed JSONs, library changes must be bit-identical (or pinned to the Fit) with no CPU regression, and the mesh CPU phase evaluates the `interferometer_numba_nnz_per_source_max` gate against the measured numba/FFT crossover.
**Verdict:** MGE: the W~ route (lever 1) and the real-valued scatter (lever 3) shipped; mixed precision gives no gain. Mesh A100: certified solver −33 % at alma is opt-in only. Mesh CPU: `cached_property` F/D shipped; crossover nnz/col ≈ 66 Delaunay / ≈ 72 rectangular. The fixed gate of 60 is superseded: PyAutoArray#632 routes by a cost model (predicted t_fft vs t_dc), and on the Aris masks (routing and log-det re-measured, timings carried from the prior run) it picks the faster arm in all six cells, where the fixed gate misrouted one. The 512 pairs are the model's own calibration points and the 125 FFT arm is a JAX proxy. The model's extent-dependent crossover answers the numba kill-gate question ([route-gate phase 2](interferometer_route_gate_aris_2026_10.md)).
**Headline:** A100 alma MGE 939.6 ms (chunked; dense path OOMs asking 61.4 GiB) → 2.69 ms on the W~ route; jvla 24.2 s → 16.7 ms.
**Library PRs:** PyAutoArray#576 (+ PyAutoGalaxy#629, PyAutoLens#750), PyAutoArray#578 (released 2026.9.27.1); PyAutoArray#540, PyAutoArray#541, PyAutoArray#544, PyAutoArray#545 (released 2026.9.11.1; library speed-ups with no profiling note, see the 2026-09-27 journal entry); PyAutoArray#582 (released 2026.9.27.2); PyAutoArray#630, #631, #632 (Aris follow-ups, merged 2026-10-15, unreleased).
**Profiling PRs:** #312, #313, #319, #324, #328, #333; phase 3 #352 (merged 2026-09-28); phase 4 #358 (merged 2026-09-30); last decision-matrix cell under issue #369.
**Ledger:** [interferometer_likelihood_decision_matrix_2026_09.md](interferometer_likelihood_decision_matrix_2026_09.md) (phase 4, the user-facing matrix), [interferometer_mge_breakdown_2026_09.md](mge/interferometer_mge_breakdown_2026_09.md), [interferometer_mesh_a100_breakdown_2026_09.md](interferometer_mesh_a100_breakdown_2026_09.md), [interferometer_mesh_cpu_breakdown_2026_09.md](interferometer_mesh_cpu_breakdown_2026_09.md); older context [numba_interferometer_verdict.md](numba_interferometer_verdict.md); Aris route gate [interferometer_route_gate_aris_2026_10.md](interferometer_route_gate_aris_2026_10.md) (#419); Aris symmetric `direct_conv` kernel [interferometer_direct_conv_symmetric_2026_10.md](interferometer_direct_conv_symmetric_2026_10.md) (PyAutoArray#635); A100 FFT size levers [interferometer_w_tilde_fft_size_levers_2026_10.md](interferometer_w_tilde_fft_size_levers_2026_10.md) (#429).
**Pulse knowledge:** [task `interferometer_large_mask_aris` (Knowledge section)](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/interferometer_large_mask_aris.md); decisions [adaptsplit-zero-weight-delaunay-only](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/adaptsplit-zero-weight-delaunay-only.md), [interferometer-route-gate-cost-model](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-route-gate-cost-model.md).
**Mind contract:** epic `interferometer-likelihood-campaign` (named in the records; not in `epics.md`); phase map `draft/research/autolens_profiling/interferometer_mesh_breakdown_numba_cpu_decision_matrix.md`.
**Next:** phase 4 has shipped: #358 merged, and the last cell (CPU rect alma_high r5.0, RAL 375978_3) was filled under issue #369. Every decision-matrix cell is now measured, blocked or not measured, with a citation; none is pending. Remaining follow-ups are separate PyAutoPulse tasks: streaming scaling, the fixed-mapper curvature preload, the nnls memo guard and the W~ FFT-size levers.

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
| 3/3 p3 radius sweep | 2026-09-28 | A100 fp64 rows at mask r2.0 / r5.0 beside the r3.5 baseline | not recorded (witness: sparse class, step sum within 10 %) | F follows the extent, ~0.8–1.2 µs per extent pixel; alma_high r5.0 190.87 / 198.55 ms, no sparse OOM | 366895, 366896 (tasks 2–5), 366907, 366908 | #352 (issue #348) |
| 3/3 p4 decision matrix | 2026-09-30 | Which interferometer likelihood path and device, by N_vis × mask × source? | witness: numba arm on `InversionInterferometerSparseNumba`, numba vs FFT ≤ 0.5 nats, step sum within 10 %, CPU vs A100 ≤ 1e-3 nats | matrix + 5-rule draft; per-call cost follows masked pixels, not N_vis; A100 W~ wins 8–100× on meshes; MGE always W~; last cell rect alma_high r5.0 filled 2026-10-04: CPU vs A100 1.59e-3 nats, accepted | 375977–375984 | #358 (issue #356); last cell issue #369 |
| Route gate p2 (Aris) | 2026-10-15 | Does the PyAutoArray#632 cost model route the Aris 125×125 / 512×512 cells to the faster arm, and do the numba and JAX log-dets agree after #631? | route matches the faster measured arm; log-det numba vs JAX at one fixed vector | cost model routes 6/6 cells to the faster measured arm (512 pairs in-sample, 125 against the JAX FFT proxy; 512 at 2,577 vertices: FFT → `direct_conv`, 1.29× faster); log-det Δ ≤ 4.4e-8 (125 vec0: −118.76 → 3.9e-8); timings carried from the prior laptop run | none (laptop) | #419 PR (library: PyAutoArray#632, #631, #630; PyAutoPulse#62) |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#576 | W~ route for MGE-only interferometer fits (closes issue PyAutoArray#575) | `1bf641e4` | 2026.9.27.1 |
| PyAutoGalaxy#629 | W~ route plumbing (PyAutoGalaxy side) | `da84468a` | 2026.9.27.1 |
| PyAutoLens#750 | W~ route plumbing (PyAutoLens side) | `e58715e9` | 2026.9.27.1 |
| PyAutoArray#578 | real scatter in `transform_mapping_matrix` (closes issue PyAutoArray#577) | `14d63360` | 2026.9.27.1 |
| PyAutoArray#582 | `cached_property` F / D on the interferometer sparse inversions (issue PyAutoArray#581) | `e281abf3` | 2026.9.27.2 |
| PyAutoArray#540 | `apply_operator` via exact `rfft2`/`irfft2` (issue PyAutoArray#538) | `7a4cb700` | 2026.9.11.1 |
| PyAutoArray#541 | preload built as a type-1 NUFFT (issue PyAutoArray#539) | `9bd76799` | 2026.9.11.1 |
| PyAutoArray#544 | NumPy/scipy sparse-operator path, no JAX on `xp=np` (issue PyAutoArray#542) | `39d3024c` | 2026.9.11.1 |
| PyAutoArray#545 | numba `direct_conv` curvature path + nnz/col gate 60 (issue PyAutoArray#543) | `35aa681f` | 2026.9.11.1 |

`PyAutoArray#582` cut every NumPy full call to 0.51–0.88x of its phase-1 value: phase-1 rows
computed F twice and D four times per `figure_of_merit` (ledger, production solver settings); the
library's evaluation-count tests went red at F=2, D=2 (positive-negative solver of the test
config) and green at 1, 1. Both counts are right; see Caveats.

## Open / parked / drafts

- Mesh phase 4 (decision matrix, issue #356): shipped in #358 (2026-09-30). Its last cell was filled under issue #369 (2026-10-04).
- `draft/research/autolens_profiling/interferometer_nnls_memo_scattered_stream_guard.md` — fnnls warm-start memo guard. The library half shipped in PyAutoArray#615 (2026.10.7.1). The #332 after-measurement ran on 2026.10.17.1 (issue #426, see the 2026-10-17 journal entry). The Delaunay iid regression is still measured at the #332 stream length. The guard's own witness, a recorded Nautilus replay, is still not run.
- W~ FFT size levers (PyAutoPulse task `interferometer_w_tilde_fft_size_levers`): measured on the A100 under #429 (2026-10-17). The pruned padded FFT passes and clears the 10 % threshold; the PyAutoArray change is drafted, not started. The coarser pixel scale fails the 0.5-nat bar.
- Route-gate constants on RAL CPU: deferred. The human closed the CPU lane on 2026-10-17 (journal, 2026-10-17).
- `draft/research/autolens_profiling/interferometer_fixed_mapper_curvature_preload.md` — fixed-mapper curvature preload.
- Certified-solver default on the interferometer sparse path went into the C2 amendment `draft/feature/autofit/certified_solver_batched_guard_c2.md`.

## Caveats

- **A round was thrown away**: RAL jobs 351055–351057 ran on nufftax 0.4.0 (below the 0.6.1 floor), which routed x64 GPU NUFFTs to fp32 Pallas (0.25 nats off at alma_high). Those JSONs are invalid and not committed; every A100 number is from the 0.6.1 re-run.
- **Above sma the A100 MGE steps are not the library path**: a single `jax.jit(FitInterferometer)` OOMs at alma and above, so steps there run on the measurement-only chunked-transform arm (agrees with the laptop fit to 1.5e-8 nats at alma).
- **Mesh A100 CPU rows** (laptop, one core, loaded) are reference only.
- **Gate move 60 → 70 rests on interpolation** inside measured brackets, not on a misrouted measured cell; machine-dependent. Superseded on 2026-10-15 by the PyAutoArray#632 cost model. The Aris 512 mask did misroute a measured cell at 60 (2,577 vertices, nnz/col 106).
- Older numba interferometer revisit found machine drift up to 2.5x between runs ([verdict](numba_interferometer_verdict.md)).
- **F/D evaluation counts before PyAutoArray#582 depend on the solver branch, not the source.** F = 2 everywhere (the cached `curvature_reg_matrix` reads F once, `fast_chi_squared` once). D = 4 per full `figure_of_merit` on the profiling rows, which run the packaged config (`use_positive_only_solver: true`, `use_edge_zeroed_pixels: true`): the pre-fix edge-zeroed fnnls branch of `reconstruction` reads `data_vector` three times (warm-start fingerprint, `[ids_to_keep]`, `.shape`) plus once in `fast_chi_squared`. D = 2 is the unit-test scenario: the #582 tests build the inversion under `test_autoarray/config/general.yaml` (`use_positive_only_solver: false`), where the positive-negative branch reads D once plus `fast_chi_squared` once. Source: PyAutoArray `879b2be1` (parent of `e281abf3`), `inversion/abstract.py` and `interferometer/abstract.py`, and the two config files. The ledger's 2 / 4 is the number that sets the measured speed-up; the record's 2 / 2 is the test's red state.
- **The 2026-09-08 library speed-ups have no RAL measurement of their own.** Their gains come from laptop or unstated-host probes (journal, 2026-09-27); the #540 `rfft2` path was never confirmed on an A100 (waived at merge).

## Journal

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.

### 2026-09-27 — library speed-ups with no profiling note

Interferometer library changes that shipped with a Mind record but no results/notes ledger of their own (the first four fell out of the [numba revisit](numba_interferometer_revisit.md) verdict).

- **PyAutoArray#540** (issue #538): `apply_operator` uses exact `rfft2`/`irfft2` instead of complex `fft2` on real input; pinned to the old route at 2.8e-14. Gain 1.27–1.61x less FFT work, laptop i9-10885H (numba revisit bake-off); A100 not measured. Merge `7a4cb700`, released 2026.9.11.1. `complete/2026/09/interferometer-apply-operator-rfft2.md`.
- **PyAutoArray#541** (issue #539): `nufft_precision_operator_from` builds the preload as a type-1 NUFFT, `eps=1e-12` default, brute-force builders kept as references. Gain alma 2101 s → 7.3 s (289x wall, 111x CPU-s), host not stated in the record (measured in autolens_profiling#229 / PR #234). Merge `9bd76799`, released 2026.9.11.1. `complete/2026/09/interferometer-preload-nufft-type1.md`.
- **Profiling-side origin of #541**: autolens_profiling#229 / PR #234 (phase 3 of the numba revisit) is a profiling task, not a library PR; it measured the builders and filed the #541 prompt. Its write-up is section 7 of [numba_interferometer_verdict.md](numba_interferometer_verdict.md). `complete/2026/09/interferometer-preload-cpu.md`.
- **PyAutoArray#544** (issue #542): `InterferometerSparseOperator` takes `xp`; a NumPy fit uses `scipy.fft` / `scipy.sparse` and never imports JAX. Gain 3.8x over the JAX-CPU route on a 40x40 / S=400 probe, host not recorded. Merge `39d3024c`, released 2026.9.11.1. `complete/2026/09/interferometer-sparse-operator-numpy-cpu-path.md`.
- **PyAutoArray#545** (issue #543): new `interferometer_numba/` package, `InversionInterferometerSparseNumba` on the `direct_conv` kernel, routed when mean nnz/col ≤ `interferometer_numba_nnz_per_source_max` (60); changed the default `xp=np` route. No measurement recorded for the library PR (it cites the prototype's 2–7x); the in-situ check is this page's mesh CPU phases 1–2. Merge `35aa681f`, released 2026.9.11.1. `complete/2026/09/interferometer-numba-cpu-direct-conv.md`.
- **PyAutoArray#582** (issue #581): `data_vector`, `curvature_matrix`, `curvature_matrix_diag` `@cached_property` on the sparse, numba and mapping interferometer inversions. Gain laptop sma Delaunay numba 492 → 343 ms, NumPy FFT 1783 → 923 ms (host not recorded); the RAL after-measurement (0.51–0.88x) is in the mesh CPU ledger. Merge `e281abf3`, released 2026.9.27.2. `complete/2026/09/interferometer-sparse-cache.md`.

### 2026-09-27 — PyAutoArray#582 evaluation counts settled

The ledger's "F twice, D four times" and the Mind record's "red at F=2, D=2" describe different runs: the full `figure_of_merit` under the packaged positive-only + edge-zeroed solver (profiling rows) versus the #582 unit tests under the test config's positive-negative solver. Read from the pre-fix source at PyAutoArray `879b2be1` and the test diff of `e281abf3`; both counts reconcile, and both are 1 / 1 after the fix. Also filled: PyAutoGalaxy#629 and PyAutoLens#750 released 2026.9.27.1 (merges `da84468a`, `e58715e9`).

### 2026-09-28 — phase 3: A100 mask-radius sweep

Issue #348. There are 12 new A100 fp64 rows (sma / alma / alma_high × Delaunay-1500 / rect 39² × mask r2.0 / r5.0), on RAL jobs 366895 and 366896 (tasks 2–5) plus the sma resubmits 366907 and 366908. The r3.5 column is the committed #324 baseline on the older mirror revisions; the ledger states the mix. The full record is in [the ledger section](interferometer_mesh_a100_breakdown_2026_09.md#mask-radius-sweep--a100-fp64-phase-3-348).

- The radius moves the cost through F alone. From r2.0 to r5.0 (6.25× the extent area), F grows 4.9–7.8× while the solve stays at 17.6–26.6 ms. The full JIT goes alma Delaunay 39.05 → 70.24 ms and alma_high Delaunay 54.10 → 190.87 ms (rect 48.87 → 198.55 ms).
- F costs ~0.8–1.2 µs per extent pixel at every instrument. alma r5.0 (200², 1M visibilities, F 37.68 ms) costs more than alma_high r2.0 (160², 5M visibilities, F 20.74 ms).
- No sparse leg OOMs, including alma_high r5.0 at vmap b16. alma dense OOMs at every radius, because T is set by N_vis × S. Step sum / full JIT is 1.002–1.051.
- For phase 4: only alma has RAL CPU radius rows (r2.0 / 4.25 / 5.0 / 6.0). sma and alma_high CPU rows at r2.0 / r5.0 must be added before the CPU-vs-A100 witness.

### 2026-09-30 — phase 4: the decision matrix

Issue #356. The [decision-matrix note](interferometer_likelihood_decision_matrix_2026_09.md) puts every committed interferometer row on one grid: N_vis (sma 190 / sdp81 1.08e5 / alma 1e6 / alma_high 5e6 / jvla 2.5e7) × mask r2.0 / 3.5 / 5.0 × source (MGE-20, Delaunay-1500, rect 39²) × path (CPU numba, CPU NumPy FFT, CPU JAX, A100 W~ sparse, A100 dense). Every cell is filled or listed as blocked / not measured with a citation.

- New: RAL CPU radius-gap cells at sma and alma_high × r2.0 / r5.0 (arrays 375977 Delaunay, 375978 rect) and an **sdp81** preset — the real SDP.81 uv coverage (108,384 visibilities) on the alma grid with the simulated lens — with CPU (375979 / 375980) and A100 (375981–375984: Delaunay, rect, MGE dense, MGE W~) rows.
- The new cells ran on the library mains from a private RAL clone (`PYAUTO_LIB_BASE`), because the shared mirror was behind main and in use by another campaign.
- Headline: per-call cost follows the masked-pixel count, not N_vis (alma r5.0 at 1M visibilities costs more than alma_high r2.0 at 5M, on CPU and A100); the packaged numba gate of 60 routes the measured CPU cells to the faster arm; the A100 W~ path is 8–100× faster than the best single-thread CPU arm on meshes.
- One pre-existing cell misses the new CPU-vs-A100 1e-3-nat bar: alma_high r3.5 rect, 1.5e-3 nats (PDIP vs fnnls on the edge-zeroed subset; 330× inside the 0.5-nat evidence bar).

### 2026-10-04 — phase 4: the last cell

This entry is for issue #369. RAL 375978_3, the CPU rect 39² cell at alma_high r5.0, finished as `COMPLETED` in 01:11:36 on `euclid-ral-gpu-1`. Its `.err` has 0 Tracebacks, 0 `RESOURCE_EXHAUSTED` and 0 float32 lines. Its JSON is committed as `alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json`.
- The numba arm runs on `InversionInterferometerSparseNumba`.
- numba and FFT agree to 7.5e-9 nats.
- `source_revisions` match the private clone: Nerves `1ec1c82`, Fit `b13169e2`, Array `7a89e19a`, Galaxy `4c834ced`, Lens `efd13c4c`.
- Per call, numba takes 75.3 s and NumPy FFT 17.6 s. nnz/col is 330.5, so the gate routes to FFT. The A100 takes 198.6 ms, which is 89×. Rule 4's 43–114× alma_high range holds.
- **CPU vs A100 is 1.59e-3 nats, which misses the 1e-3 bar.** The CPU value is −60244101.501766354 (fnnls, `7a89e19a`). The A100 value is −60244101.503356226 (PDIP, pre-#595 `9428eca2`).
- **The human accepted the gap on 2026-10-04.** It matches the r3.5 precedent: a solver/revision mismatch, not precision. It is 310× inside the 0.5-nat bar.

Every decision-matrix cell is now filled.

### 2026-10-15 — route gate phase 2: the Aris table under the cost model

Issue #419; record [interferometer_route_gate_aris_2026_10.md](interferometer_route_gate_aris_2026_10.md).
- PyAutoArray#632 (`17caf228`) replaced the fixed nnz/col gate of 60 with the cost model `t_fft = c_fft·S·pocketfft-work(2ny×2nx)`, `t_dc = c_dc·nnz·ny·nx`. The packaged default is now the cost model.
- Under default settings it routes the 125×125 cells (extent 83²; 1,405 / 2,947 / 5,526 vertices) to `direct_conv`. It routes the 512×512 cells (extent 341²) to FFT at 1,459 vertices and to `direct_conv` at 2,577 and 5,746: the faster measured arm in all six. The old gate sent 2,577 vertices (nnz/col 106) to the slower FFT arm. The 512 timings are among the model's calibration points, so that agreement is in-sample. The 125 FFT arm was only timed under JAX jit.
- **The numba kill-gate question is answered by the cost model, not a retuned constant.** The *modelled* crossover is ~309 nnz/col at 83² and ~166 at 341². The measured cells alone would also fit one constant of about 150.
- The campaign's owed log-det after-measurement: numba vs JAX `log_det_regularization_matrix_term` agree to ≤ 4.4e-8 at all six cells. At the 125 original vector the pair went from 2443.22 / 2561.98 to 2468.4463 on both (PyAutoArray#631).
- The laptop run recorded routing and log-det only. s/call is carried from the prior laptop measurement, not re-timed, because the host was shared.

### 2026-10-17 — symmetric row-restricted `direct_conv` kernel (PyAutoArray#635, lever L2)

Record: [interferometer_direct_conv_symmetric_2026_10.md](interferometer_direct_conv_symmetric_2026_10.md). Library PR: PyAutoArray#636 (pending release, not merged). Campaign: PyAutoPulse#62.

- **The kernel.** With serial numba, `direct_conv` now computes only the ranked upper triangle of F, restricted to the extent rows that later-ranked columns span and to each row's mask x-range. It then mirrors the rest. Computed entries are bit-identical to the full kernel.
- **The gate.** The kernel runs when the geometry work fraction f ≤ `interferometer_direct_conv_symmetric_max_work_fraction`: default 0.7, `0` turns it off. The cost model is t ∝ κf with κ = 1.32. The route model scales `t_dc` the same way.
- **Real Aris data** (SPT2319-55 512×512; RAL `euclid-ral-compute-1`, `ral`, exclusive, 1 thread; jobs 412490–412493): the curvature step speeds up **1.49× / 1.53× / 1.58×** at 1,459 / 2,577 / 5,746 vertices (f 0.511 / 0.504 / 0.495). log L agrees to ≤ 2e-12 relative. The 5,746-vertex log-det difference of 0.0071 nats matches the full kernel's own lower-vs-upper Cholesky round-off of 0.0077 nats at cond 7.2e13.
- **Synthetic Aris-512 geometry:** 1.83× / 1.86× / 1.90× at f ≈ 0.41.
- **Tall columns.** Multiple imaging makes the real Delaunay columns tall: the median spans 90 of 341 rows, against 15 synthetic. That is why real data gets about 1.5× rather than about 1.9×, and why the human chose ascending-`ymin` ordering (f 0.587 → 0.511) and a gate of 0.7 rather than 0.5 (2026-10-16).
- **At 1,459 vertices the FFT route measures 15.32 s on the same node**, so the switch to `direct_conv` there gains only about 1.02×. On RAL, FFT/full `direct_conv` is 0.68, against 0.89 predicted by the laptop-calibrated #632 constants. Refitting those constants on cluster nodes is left to the FFT size-lever phase.

### 2026-10-17 — W~ FFT size levers on the A100, and the route constants left open

Issue #429; record [interferometer_w_tilde_fft_size_levers_2026_10.md](interferometer_w_tilde_fft_size_levers_2026_10.md); witness `jvla/delaunay_hpc_a100_fp64_fft_prune.json`. RAL job 412515 ran on `euclid-ral-gpu-1` (A100 80GB, driver 610.57.04) with JAX 0.10.2, fp64, on a private 2026.10.17.1 stack.
- **Pruned padded FFT: passes.** A separable transform that skips the zero rows of the `(2y, 2x)` pad saves 19.4 / 22.1 / **29.7 %** of F at alma / alma_high / jvla. At jvla F goes from 499.6 to 351.2 ms.
  - The pruned F matches the library F to ≤ 1.7e-15 relative, and the figure of merit is unchanged.
  - The fused F → solve → log-det → figure-of-merit call is 1.08× / 1.18× / **1.41×** faster.
  - This clears the task's 10 %-of-F threshold, so the PyAutoArray change is drafted. The human set no library change for this phase.
- **Real-space pixel scale at jvla: fails the bar.** F gets 1.53× / 4.09× / 7.61× cheaper at 0.015 / 0.02 / 0.025″, but the log evidence moves by +1,310 / +1,807 / +2,757 nats against 0.01″. The bar was 0.5 nats.
  - Most of the shift is in `log det(F + H)`, and the mesh is rebuilt on each scale's adapt image.
  - 0.015″ gains only 1.53× because its padded FFT length 932 = 2²·233 is an awkward length.
- **Open note: route-gate constants (part A of #429), not measured.** The human closed the interferometer CPU lane on 2026-10-17, so the constants refit is deferred. The evidence in hand is from PyAutoArray#636's witness, recorded in `wiki/campaigns/interferometer/interferometer_direct_conv_symmetric_2026_10.md` on #425 (not yet merged):
  - On `euclid-ral-compute-1`, serial, FFT / full `direct_conv` measured **0.68** at the real-Aris 512 1,459-vertex cell. The laptop-calibrated #632 constants (`c_fft` 7.62e-10, `c_dc` 1.77e-09) predict **0.89**.
  - With the symmetric kernel, that cell now routes to `direct_conv` at only **~1.02×** over FFT (15.08 vs 15.32 s).
  - No route measured so far is wrong, but the margin is thin on RAL.

### 2026-10-17 — fnnls memo after-measurement on 2026.10.17.1 (issue #426)

Summary [memo_after_hpc_ral_cpu_fp64_v2026.10.17.1.json](../../../results/misc/likelihood_breakdown/interferometer/memo_after_hpc_ral_cpu_fp64_v2026.10.17.1.json); RAL job 412526.
- **Setup.** The #332 `memo` lever was re-run on the same cells (Delaunay-1500 and rect 39², alma r3.5, numba arm on one thread) and the same dataset (alma data / uv md5 identical to the RAL tree). The five libraries came from a private 2026.10.17.1 checkout, which contains PyAutoArray#615's scattered-stream back-off. The job ran on partition `ral`, node `euclid-ral-compute-10-2` (load about 10–11 on 236 cores from other users' arrays). #332 ran on `euclid-ral-gpu-2` CPUs, so the host class differs. Each cell ran as six independent rounds at two iid lengths: `n6` (the exact #332 flags, 4 timed iid solves) and `n17` (16 iid instances, 15 timed). The walk was 16 instances (15 timed). The headline is median(on) / median(off) with the paired whole-round bootstrap 90 % interval (#407, ≥ 5 rounds per #410); the #332 mean ratio is shown beside it.
- **Harness fixes.** `_lever_memo` now clears the memo with `nnls_memo.memo_clear()`. That also resets #615's back-off state, which clearing the dict alone would carry from the iid pass into the walk pass. The harness also reads the packaged numba gate when it is `None` (the PyAutoArray#632 cost model). The arms pass explicit gates, so routing is unchanged.

| cell | stream | timed solves × rounds | median on / off [90 % CI] | #332 mean estimator, median of rounds (range) | #332 (2026.8.17.1) | slow-seed on-solves |
|---|---|---|---|---|---|---|
| Delaunay | iid, n6 (#332 flags) | 4 × 6 | **1.918 [1.887, 1.925]** | 1.96 (1.95–2.67) | 2.17 | 12 / 24 |
| Delaunay | iid, n17 | 15 × 6 | 1.018 [1.013, 1.023] | 1.48 (1.47–1.50) | — | 24 / 90 |
| Delaunay | local walk | 15 × 6 | 0.460 [0.455, 0.463] (n6 run) / 0.458 [0.452, 0.461] (n17 run) | 0.49 | 0.68 | 0 / 90 |
| rect 39² | iid, n6 | 4 × 6 | 0.887 [0.884, 0.891] | 0.92 | 1.08 | 0 / 24 |
| rect 39² | iid, n17 | 15 × 6 | 0.987 [0.983, 0.990] | 0.96 | — | 3 / 90 |
| rect 39² | local walk | 15 × 6 | 0.150 [0.149, 0.154] / 0.151 [0.151, 0.153] | 0.15 | 0.13 | 0 / 90 |

A slow-seed on-solve is one slower than 1.5× its round's off median. The figure of merit is identical on and off (Δ 0 nat) in every pass.

- **Reading.** On 2026.10.17.1 the Delaunay iid regression is still present at #332's stream length: median 1.92×, and the #332 estimator gives 1.96× against 2.17×. Its shape has not changed. Every other on-solve pays a bad seed (537 / 173 / 922 / 156 ms against an off median of about 171 ms in round 1), and the same positions are slow in every round. On the longer iid stream the bad seeds thin out after the first ten solves: slow at timed positions 1, 3, 6 and 10, none in the last five. That is consistent with #615's back-off lengthening its skip windows. The median ratio is then 1.02×, but the mean is still 1.48×, because each bad seed costs about 2.7–3× a dense solve. The rect iid stream no longer regresses (0.89–0.99×). The walk gains are kept: Delaunay 0.46× (#332: 0.68×), rect 0.15× (#332: 0.13×).
- **Limits.** The six rounds resample timing noise around one fixed iid stream (`IID_SEED`), not different streams, so the intervals are narrow and say nothing about stream-to-stream spread. The load on the node was about 10. The host class is not #332's. This is the after-measurement only. The Pulse task's witness, a recorded Nautilus evaluation sequence replayed phase by phase, has not been run, and this entry makes no claim that PyAutoArray#613 is fixed.
