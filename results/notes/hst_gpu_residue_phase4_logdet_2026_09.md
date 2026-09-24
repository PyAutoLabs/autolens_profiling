# HST GPU residue phase 4 — reusing the certified solve's factor for log det(F + λH)

## Verdict

**No lever.** Phase 4 tested reusing the certified solve's Cholesky factor for
`log det(F + λH)` on the A100, with a Schur complement over the fixed (and edge) set. The
identity is exact: all eight rows pass the pre-registered `1e-9` gate (72/72 pins). But no
candidate is faster than the library's own second dense Cholesky, on either mesh:

- **Delaunay N=1500.** The Schur candidates are *slower* than the unmodified library route in
  the same task, by 0.27 ms (`schur_k32`), 0.31 ms (`schur_k64`) and 0.45 ms (`schur_k256`) on
  the interleaved median. The slowdown grows with `k_max`. All three timed calls took the Schur
  branch (fiducial |Z| = 4).
- **Rectangular (1521 slots, 152 edge-zeroed).** The fiducial |Z| is 232 (80 fixed plus 152 edge),
  so `schur_k32` and `schur_k64` (184 and 216 slots) overflowed. Their timed call took the dense
  `lax.cond` branch and ties the library (+0.09 ms, inside noise). `schur_k256` (408 slots) took
  the Schur branch and is 0.38 ms slower.

The pre-registered lever threshold is ≥ 0.5 ms saving on **both** the `jit_profile` and the
interleaved-median basis, within one task. No candidate reaches it with either sign. The best
saving on any row is +0.09 ms, the same size as the control rows' own noise. **No PyAutoArray
prompt is filed.**

The reason is structural, and it shows in the trace (see
[Why there is no lever](#why-there-is-no-lever)). Reusing the factor still needs one
triangular solve against the full 1500×1500 factor. On the A100 that solve costs 0.97-1.05 ms
at every `k` from 32 to 256. The dense cuSOLVER Cholesky it replaces costs 0.90 ms.

## Experiment

- Job: RAL SLURM array `350651`, tasks 0-7, one A100 80GB PCIe each. Submitted 2026-09-24
  14:46:22 UTC. The Delaunay tasks (0-3) ran on `euclid-ral-gpu-1` and the rectangular tasks
  (4-7) on `euclid-ral-gpu-2`. Every task was `COMPLETED 0:0` in 77-89 s. Every footer shows
  `AUTOTUNE_ENTRIES count=0` and `CELL_EXIT=0`. No `.out` or `.err` contains a traceback. Each
  `.err` holds only the benign `mask_2d_util.py:564` padding warning.
- Profiling source: `feature/hst-gpu-residue-p4` @ `a621160`. Every JSON's `cell_source_sha256`
  (`8046c44e…`) equals the sha256 of `fixed_light_trace.py` at that commit.
- Library revisions: PyAutoNerves `1fa613aa`, PyAutoFit `a7368401`, **PyAutoArray `7fa8d271`**,
  PyAutoGalaxy `70a61e26`, PyAutoLens `86054bbc`. These are the canonical mains and `origin/main`
  on 2026-09-24. Three RAL checkouts are flagged `dirty: true` in the JSONs. In each case the only
  change is one untracked emacs autosave file (`#…#`); no tracked file is modified. The cell's
  stage anchors cite PyAutoArray `681938ae`. The range `681938ae..7fa8d271` touches nothing under
  `autoarray/inversion/`, `autoarray/fit/` or `autoarray/util/jax_active_set.py`. The dataset
  sha256s are identical to those in phases 2 and 3.
- Likelihood: HST imaging, fp64, fixed lens light, dense inversion, border relocator on. The
  solver is the **library** certified active-set solver (PyAutoArray #566), selected through
  `al.Settings` (`positive_only_solver=certified`, `certified_fallback=pdip`, packaged
  `certified_pass_budget=16`), under the scalar-`jit` program that phase B adopted. No harness
  solver is used in this phase.
- The candidate is a harness rebind
  (`scripts/misc/likelihood_breakdown/logdet_reuse_injection.py`), active only inside a scoped
  context. It replaces two library entry points: `jax_active_set.solve_certified`, with a
  bit-identical wrapper that keeps the final masked Cholesky factor, and
  `AbstractInversion.log_det_curvature_reg_matrix_term`. The candidate computes
  `log det M = 2 Σ log diag L + log det S + Σ log diag M` in the Jacobi-scaled space, where `S` is
  the Schur complement over `Z` (the fixed set plus the unsolved edge indices), padded to a static
  `k_slots`. If `|Z| > k_slots`, a `lax.cond` falls back to the dense route.
- Injection counts confirm the candidate ran in every non-control row: `route_d_jax = 1`,
  `traced_program_jax = 1`, `report_program jax = 1`, with zero delegated. The HLO census is `ok`
  on every row. Candidates compile 8 Cholesky factorizations against control's 7: the new small
  Schur Cholesky is added, and the dense one is kept for the fallback branch.
- Fresh JAX compilation cache per task, with zero autotune entries at start.

## Numerical gate

The gate and the 0.5 ms lever rule were **pre-registered** at `71e3f21` (2026-09-24 14:20:27
UTC). The amendment followed at `5f6f8c7` (14:22:49 UTC). It added `schur_k256` and the separate
timed-point |Z| record, because the RTX screen had found |Z| = 4 on the fiducial but 51-250 on
the draws. The amendment came 24 minutes before the A100 submission, and no A100 data existed
yet. The gate itself was not changed.

The gate:

- **Pins (gated).** Route d (candidate) is pinned against the **unmodified library route**: the
  same Settings and composition, compiled without the rebind, in the same process. The pin is at
  `1e-9` relative, on the fiducial and on 8 seeded draws (seed 0). It applies to every task,
  control included.
- **Trace checks (gated).** |reconciliation| ≤ 5 % and zero unjoined device time.
- **JAX path (gated).** Every non-control candidate must be taken on the JAX path.
- **Recorded, not gated.** Every row against route b (library PDIP). Phases 2 and 3 placed a
  2.0-2.6e-9 certified-vs-PDIP residual in the solver path, and this phase does not touch it.

| mesh | candidate | pins | fiducial rel | max rel (9 pins) | max nats | max rel vs route b | recon % | unjoined ms | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| delaunay | `control` | 9/9 | 2.46e-11 | 2.08e-10 | 3.8e-6 | 1.80e-10 | −0.53 | 0 | PASS |
| delaunay | `schur_k32` | 9/9 | 6.89e-11 | 1.60e-10 | 2.1e-6 | 1.28e-10 | −0.51 | 0 | PASS |
| delaunay | `schur_k64` | 9/9 | 8.23e-12 | 3.66e-10 | 3.3e-6 | 3.97e-10 | −0.60 | 0 | PASS |
| delaunay | `schur_k256` | 9/9 | 7.71e-11 | 2.46e-10 | 4.5e-6 | 1.94e-10 | −0.47 | 0 | PASS |
| rectangular | `control` | 9/9 | 0 | 0 | 0 | 4.73e-13 | −0.49 | 0 | PASS |
| rectangular | `schur_k32` | 9/9 | 0 | 0 | 0 | 4.73e-13 | −0.46 | 0 | PASS |
| rectangular | `schur_k64` | 9/9 | 0 | 0 | 0 | 4.73e-13 | −0.43 | 0 | PASS |
| rectangular | `schur_k256` | 9/9 | 0 | 1.4e-16 | 3.6e-12 | 4.73e-13 | −0.47 | 0 | PASS |

The pins are exactly 0 on rectangular `schur_k32` and `schur_k64` because every one of their
nine rows overflowed into the dense branch, which computes what the library computes. On
`schur_k256`, eight rows took the dense branch; draw 0 (|Z| = 366) took Schur and differs by
1.4e-16. The Delaunay control pin is a second compilation of the same program, and it still
differs by up to 2.1e-10. That is the fp64 reproducibility floor of two compilations on this
mesh, and the Schur rows sit at the same level. Every row is also within `1e-9` of route b, so
the recorded PDIP pin would not have changed the verdict.

## Matched timing table

All comparisons are within a single task: the candidate against the unmodified library route
compiled in the same process. Phase B measured ~3 % differences between nodes, so rows are never
compared across tasks. **Saving = library − candidate** (positive means the candidate is faster).

- **jit_profile:** 10 steady calls of each program.
- **Interleaved:** 7 alternating blocks of 10 calls. The median of each program's block means is
  shown, and "block range" is the spread of the per-block paired saving.
- **log-det row:** the traced `log_det_curvature_reg` stage (median of 10 calls, command buffers
  off).

| mesh | candidate | library ms | candidate ms | saving jit_profile | library median | candidate median | **saving interleaved** | block range | log-det row ms | lever |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| delaunay | `control` | 26.94 | 26.69 | +0.24 | 26.57 | 26.48 | **+0.09** | −0.03…+0.42 | 0.897 | — |
| delaunay | `schur_k32` | 26.65 | 26.73 | −0.08 | 26.43 | 26.70 | **−0.27** | −0.32…+0.05 | 1.075 | no |
| delaunay | `schur_k64` | 26.59 | 26.87 | −0.27 | 26.54 | 26.85 | **−0.31** | −0.48…−0.20 | 1.142 | no |
| delaunay | `schur_k256` | 26.42 | 26.90 | −0.48 | 26.35 | 26.79 | **−0.45** | −0.68…−0.36 | 1.284 | no |
| rectangular | `control` | 26.92 | 26.93 | −0.02 | 26.94 | 26.96 | **−0.03** | −0.12…+0.12 | 0.932 | — |
| rectangular | `schur_k32` | 26.85 | 26.99 | −0.14 | 26.89 | 26.80 | **+0.09** | +0.03…+0.17 | 0.966 | no |
| rectangular | `schur_k64` | 26.88 | 26.93 | −0.05 | 26.92 | 26.83 | **+0.09** | −0.03…+0.40 | 0.970 | no |
| rectangular | `schur_k256` | 26.72 | 27.29 | −0.57 | 26.84 | 27.22 | **−0.38** | −0.59…+0.47 | 1.368 | no |

**Noise floor.** A control row runs the same program in both arms, so its "saving" is pure
noise. The controls read +0.24 / −0.02 ms on `jit_profile` and +0.09 / −0.03 ms on the
interleaved median. A single block can move by up to 0.42 ms. Measured this way, the 0.5 ms
threshold sits about 2× above the noise, and every candidate reading is inside the noise or
negative.

Other rows, all in the same tasks:

- **Route b (library PDIP):** 50.64-50.83 ms on Delaunay and 38.86-39.08 ms on rectangular,
  against 26.3-26.9 ms for the library certified route.
- **Traced executable:** 26.5-28.1 ms with command buffers on or off.
- **Certified-solve row:** unchanged by the rebind (5.03-5.06 ms Delaunay, 10.68-10.73 ms
  rectangular), so keeping the factor costs nothing measurable.
- **Compile time (route d):** 4.50-5.23 s on Delaunay; `schur_k256` is +0.6 s over its own
  library control (4.62 s). 4.02-4.31 s on rectangular. One anomaly: the rectangular control's
  second compilation of its own program took 0.19 s, apparently a hit in the fresh per-task cache
  that route d had just populated. It is recorded and does not affect any timed number.
- **Peak device memory:** 1.185 GB (Delaunay) and 1.201 GB (rectangular) on every row. The rebind
  adds no measurable memory.
- **Trace checks:** profiler overhead was 3.6-8.0 % of the untraced wall (not gated). The host
  qhull callback and device idle (7.3-8.3 ms Delaunay, 2.1-2.2 ms rectangular) are not affected.

The Delaunay library route reproduces within the ~3 % node-to-node band quoted in phase B.

## |Z| and the branch the timed call took

In every task, all whole-call timings and trace rows come from the **fiducial** call. The draws'
|Z| values come from a separately compiled, untimed report program. All 72 rows were certified
by the library solver (`uncertified_count = 0`).

| mesh | candidate | k_slots | fiducial \|Z\| | **timed branch** | draws \|Z\| (seed 0, 8 draws) | draws overflowing |
|---|---|---:|---:|---|---|---:|
| delaunay | `schur_k32` | 32 | 4 | **schur** | 51, 199, 250, 244, 88, 67, 242, 205 | 8/8 |
| delaunay | `schur_k64` | 64 | 4 | **schur** | same | 7/8 |
| delaunay | `schur_k256` | 256 | 4 | **schur** | same | 0/8 |
| rectangular | `schur_k32` | 32 + 152 = 184 | 232 | **dense (overflow)** | 366, 725, 767, 894, 517, 494, 943, 793 | 8/8 |
| rectangular | `schur_k64` | 64 + 152 = 216 | 232 | **dense (overflow)** | same | 8/8 |
| rectangular | `schur_k256` | 256 + 152 = 408 | 232 | **schur** | same | 7/8 |

The fiducial is a near-best-fit point, where very few pixels are pinned at zero. The draws sit
further out, where 51-250 pixels (Delaunay) or 214-791 (rectangular, excluding the 152 edge
slots) are fixed. A sampler spends most of its evaluations at points like the draws. So even a
Schur path that won at the fiducial would take the dense fallback on most production evaluations
unless `k_slots` covered |Z| ≈ 250-950. Its triangular-solve and Schur-Cholesky cost grows with
`k_slots`. This is a second, independent reason why no `k` choice produces a lever.

## Why there is no lever

The traced log-det stage row explains the whole-call results. Its top kernels:

| row | log-det row ms | top kernel (ms, shape) | second kernel |
|---|---:|---|---|
| delaunay `control` | 0.897 | `cholesky` 0.896, f64[1500,1500] | — |
| delaunay `schur_k32` | 1.075 | `triangular-solve` 0.968, f64[1500,32] | `cholesky` 0.040, f64[32,32] |
| delaunay `schur_k64` | 1.142 | `triangular-solve` 1.029, f64[1500,64] | `cholesky` 0.043, f64[64,64] |
| delaunay `schur_k256` | 1.284 | `triangular-solve` 1.049, f64[1500,256] | `cholesky` 0.153, f64[256,256] |
| rectangular `control` | 0.932 | `cholesky` 0.932, f64[1521,1521] | — |
| rectangular `schur_k32` / `k64` (dense branch) | 0.966 / 0.970 | `cholesky` 0.930 / 0.935, f64[1521,1521] | `input_transpose_fusion` 0.026 |
| rectangular `schur_k256` | 1.368 | `triangular-solve` 0.994, f64[1369,408] | `cholesky` 0.259, f64[408,408] |

- **The factor is reused, but the triangular solve is not cheap.** The Schur complement needs
  `L⁻¹ B`, where `B` holds the Z columns and `L` is the 1500×1500 factor. On the A100 this
  `trsm` costs about 1 ms, whether `B` has 32 or 256 columns (0.97 → 1.05 ms). The near-flat
  cost in `k` suggests it is bound by sequential latency across the triangle rather than by FLOPs
  (an inference from the kernel times, not a separate measurement). That is slightly *more* than the whole
  dense cuSOLVER factorization it was meant to save (0.90 ms). The small Schur Cholesky
  (0.04-0.26 ms) and the gathers then add to it.
- **The stage change tracks the whole-call change.** On Delaunay, the log-det row rises by +0.18,
  +0.25 and +0.39 ms, and the interleaved call changes by −0.27, −0.31 and −0.45 ms. On
  rectangular `schur_k256`, the row rises by +0.44 ms and the call changes by −0.38 ms. On the
  two overflowed rectangular rows, the `lax.cond` adds only +0.03-0.04 ms (a 0.026 ms transpose
  fusion), and the call ties the library.
- **The ceiling was never above threshold.** At best the dense 0.90 ms Cholesky disappears
  entirely, and even that assumes |Z| = 0 and no triangular solve. The measured cost of the
  reuse path exceeds that ceiling at every `k`. No `k_max` tuning can get past a ~1 ms `trsm`
  floor.

Phase 1 measured this term at 0.890 ms (`cholesky.110`). The control row here measures 0.897 ms:
the target reproduced, and the reuse path simply costs more than the target.

## What this means for the campaign

This section records the state of the campaign; it does not make a decision. Phase 4 closes the
last fp64 lever listed in the campaign map built from phase 1's single-call trace:

- the qhull callback idle, parked behind the batching-reproducibility study (phase 2);
- the `A.T A` F GEMM, closed as the fp64 dense floor;
- the mixed fusions, closed as the attribution's resolution limit;
- the PSF convolution of the mapping-matrix cube, closed with no fp64 lever (phase 3);
- **the second Cholesky for log det(F + λH), closed with no lever (this phase).**

**The fp64 levers in the campaign map are exhausted.** What remains is not a profiling
question under the campaign's fp64 constraint:

- the **fp32 precision policy** (phase 3's diagnostic rows: 7-13 % of the call for ~1e-3 nats
  per evaluation), which is a human decision;
- **batched / callback work** (vmap batching policy and batch-aware host callbacks), which is
  filed and tracked elsewhere.

## Provenance and historical separation

The eight A100 JSON/PNG pairs are
`results/breakdown/imaging/fixed_light_trace_<mesh>_logdet_<candidate>_hpc_a100_fp64_fixed_light_trace.{json,png}`.
Their sha256s, the SLURM per-task state (`sacct`, including MaxRSS), the footers and the gate
decision are in [`hst_gpu_residue_phase4_job350651.json`](hst_gpu_residue_phase4_job350651.json).
The `.out` and `.err` files stay on RAL, with local copies in the gitignored
`output/ral_job350651/`.

The RTX 2060 rows are Delaunay-only screening, committed at `a621160`. They checked the pins
(9/9, ≤ 3.1e-10), the reconciliation, the injection counts and the exit codes. Their timings were
contaminated by a concurrent GPU job and are not quoted, and GeForce fp64 rows are never ranked.
The artifacts are
[`fixed_light_trace_delaunay_logdet_<candidate>_local_rtx2060_fp64_fixed_light_trace.json`](../breakdown/imaging/fixed_light_trace_delaunay_logdet_control_local_rtx2060_fp64_fixed_light_trace.json)
(control linked; the three siblings share the pattern). The `schur_k64` RTX row predates the
timed-point record.

The notes for phases 1-3 are unchanged. No PyAutoArray, PyAutoLens or PyAutoFit source was
changed.
