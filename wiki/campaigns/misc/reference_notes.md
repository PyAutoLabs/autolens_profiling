# Reference notes

**Status:** complete
**Question:** n/a — standing reference notes that belong to no single campaign (solver ledgers, design decisions, cell definitions, earlier campaign indexes).
**Pre-registered rule:** n/a
**Verdict:** n/a
**Headline:** n/a
**Library PRs:** n/a
**Profiling PRs:** n/a
**Ledger:** the notes listed below
**Mind contract:** n/a
**Next:** none; each note below says whether it is live, superseded or frozen

These are the notes a campaign page cites for a rule, a cell definition or an old number, but
that no single campaign owns. Each entry says what the note records, its date and status, the
figure or decision it is cited for, and which campaign page it feeds. Read the note itself before
quoting a number: several carry their own supersession banners.

## The notes

- [nnls_solver_ledger.md](../../results/notes/nnls_solver_ledger.md) — NNLS positive-only solver ledger (PyAutoArray#369 knobs, #370 BPP/ADMM), closed 2026-07-09, on real extracted HST rect n=1581 / Delaunay n=1560 systems.
  Cited for: PDIP at ~20 iterations × fresh Cholesky is near-optimal at cond ~1e10; the only lever is the default-off `Settings(nnls_solver_tol=1e-6)` (~15–20 % of the solve, Δlog_ev ~1e-8); BPP, ADMM, warm start and fp32 all lose.
  Its A100 rows (job 330046) are still listed as pending. Background for [Certified positive solver](certified_positive_solver.md).
- [sparse_vs_dense_inversion_path.md](../../results/notes/sparse_vs_dense_inversion_path.md) — sparse (w-tilde) vs dense inversion path, issue #44: CPU HST baselines, six A100 runtime jobs (323017–323022) and NSS/Nautilus production validation.
  Cited for: CPU sparse wins pix −41 % / Delaunay −34 %, loses MGE +51 %; on A100 sparse is ~7–10 % slower per call but uses 7–10× less VRAM per replica ("enable sparse for memory, not speed").
  A100 pixelized rows superseded 2026-09-10 by [A100 pixelized baseline](a100_pixelized_baseline.md); CPU rows and production guidance stand.
- [vram_validation_2026_07_08.md](../../results/notes/vram_validation_2026_07_08.md) — polish phase 2 VRAM-first CPU validation sweep, 2026-07-08, issue #54, before any A100 time was spent.
  Cited for: all 9 PreOptimizationTimes cells run clean end-to-end (47 s to 401 s on CPU), after three fixes: backend-aware vmap batch clamp, per-instrument pins, and pins turned from hard asserts into record-and-flag.
  Feeds [PreOptimizationTimes baseline](preoptimization_baseline.md).
- [design_lock_in.md](../../results/notes/design_lock_in.md) — design lock-in review before PreOptimizationTimes, 2026-07-08, issue #52; its task-first package layout was inverted to dataset-first on 2026-07-24 (#84), the rest stands.
  Cited for: the cell grid, the single `_profile_cli.py` surface, append-only named baselines, pins as record-and-flag (`pinned_drift` is a fault list, #261), and the CPU-usability rule (> 3600 s run or > ~1 min per call = GPU-only).
  Feeds [PreOptimizationTimes baseline](preoptimization_baseline.md) and every later cell.
- [production_representative_cells.md](../../results/notes/production_representative_cells.md) — the four CPU numba imaging cells switched to the production configuration per instrument, 2026-09-08, #235 (with #237 moving Euclid lp to `[4, 4, 2]`).
  Cited for: the pre-#235 Euclid `delaunay_numba` 1.19 s headline was a memo-self-seeded warm mean; the production Euclid cold eval is 0.239 s (laptop i9-10885H, memo off), with field-by-field provenance against Euclid DR1 job 342301 and subhalo job 342311.
  Precursor of [Imaging over-sampling](imaging_over_sampling.md).
- [preopt_breakdown_baseline.md](../../results/notes/preopt_breakdown_baseline.md) — PreOptimizationTimes likelihood-breakdown baseline, polish phase 4 (#59), laptop CPU and A100 imaging tiers complete 2026-07-10.
  Cited for: the bottleneck moves on GPU (CPU mesh cells are F-dominated at 42–48 %, A100 pixelization is NNLS at ~65 %); A100 imaging jobs 330062–330070; Delaunay A100 rows superseded 2026-09-05.
  Owned by [PreOptimizationTimes baseline](preoptimization_baseline.md).
- [point_source_defaults_campaign.md](../../results/notes/point_source_defaults_campaign.md) — point-source defaults evidence for PyAutoLens#678 phase B, A100 fp64, complete 2026-08-02 (23/25 cells; the two unfinished arms are themselves a finding).
  Cited for: tensor source-plane weighting, solved centres and all-to-all image-plane pairing (`PairRepeatSolved` mis-ranks truth by ~1.8×10⁵ log-likelihood with one image missing); gradient searches not yet competitive at cluster scale.
  Its searches-framework instrument was retired with the inference programme. Feeds no current campaign page.
- [point_source_shared_likelihood_breakdown.md](../../results/notes/point_source_shared_likelihood_breakdown.md) — the shared CPU/GPU image-plane `PointSolver` breakdown instrument, 2026-09-20, issue #291.
  Cited for: the laptop CPU fp64 reference (solved fused likelihood 62.687 ms/call, compile 17.643 s), and the rule that prefix step rows telescope to the fused likelihood and are not independent kernel timings.
  Instrument for [Point-source image-plane CPU](point_source_image_plane_cpu.md) and [Point-source A100 breakdown](point_source_gpu_breakdown.md).
- [numba_curvature_matrix_f_split.md](../../results/notes/numba_curvature_matrix_f_split.md) — numba CPU curvature matrix F sub-block split, PyAutoArray#505 (phase 1) and phase 2 (#507), measured 2026-08-28 on the laptop on the pre-#235 fiducial.
  Cited for: the dense mapper x linear-func block was 70–85 % of F; phase 1 cut it ~16× (HST eval 2.6×); phase 2 took HST rectangular 0.62 → 0.33 s (1.86×).
  Timings predate [production_representative_cells.md](../../results/notes/production_representative_cells.md). Feeds no current campaign page.
- [profiling_campaign_status_2026_09.md](../../results/notes/profiling_campaign_status_2026_09.md) — fixed-light CPU + GPU residue status index, 2026-09-18 (GPU verdicts 2026-09-23/24), superseded by [`wiki/index.md`](../index.md) for navigation, kept as the fixed-light record.
  Cited for: CPU fixed light 932.4 → 459.2 ms (2.03×, memo off) and 413.3 → 230.0 ms (1.80×, memo on; do not multiply the two); the A100 budget-7 Delaunay call is 31.64 ms, not 25.39 ms.
  Feeds [Fixed-light numba CPU](fixed_light_numba_cpu.md) and [HST GPU residue](hst_gpu_residue.md).

## Caveats

- **Laptop rows in these notes are not quotable as RAL figures**: `production_representative_cells.md`, `numba_curvature_matrix_f_split.md` and the CPU tier of `preopt_breakdown_baseline.md` are single-host laptop measurements.
- `sparse_vs_dense_inversion_path.md` dates its superseded A100 rows to 2026-07-11, but their job ids (323017–323022) are lower than the 2026-07-10 PreOptimizationTimes A100 jobs (330062+); the date is not verified.
- The `profiling_campaign_status_2026_09.md` cumulative CPU figure bridges on job 343355's control, whose JSON/PNG were never committed (the note records this).

## Journal

### 2026-09-27 — stub created

Collected the standing reference notes; summaries pending backfill.

### 2026-09-27 — backfilled (phase 2)

Each of the ten notes now has a two-to-three-line summary read from its opening sections: what it
records, date and status, the figure or decision it is cited for, and the campaign page it feeds.
All ten ledger links kept. Flagged: a date mismatch in the sparse-vs-dense A100 rows, and the
NNLS ledger's A100 rows still listed as pending. Nothing left open on this page.
