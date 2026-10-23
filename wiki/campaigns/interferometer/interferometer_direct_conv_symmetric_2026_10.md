# Interferometer `direct_conv`: the symmetric row-restricted kernel and its gate (PyAutoArray#635)

Lever L2 of the Aris large-mask campaign (PyAutoPulse#62, task `interferometer_large_mask_aris`).
Library PR: [PyAutoArray#636](https://github.com/PyAutoLabs/PyAutoArray/pull/636) (issue
[#635](https://github.com/PyAutoLabs/PyAutoArray/issues/635)). This note records what the kernel does,
the rule that decides when it runs, and the measurements behind that rule. It is written so that
autolens_assistant can choose an interferometer setup from it.

## Decision rule (for choosing a setup)

1. **Scope.** The rule applies only to the NumPy (`xp=np`) single-mapper interferometer inversion on
   the numba `direct_conv` route, **with serial numba** (`parallel: false`, one thread per process, as
   in a Nautilus run with one worker per core). The parallel numba kernel and every JAX route are
   unchanged.
2. **Work fraction.** `f` is the fraction of the full kernel's convolution work that the symmetric
   kernel still does. It is a pure geometry number, computed from the mapper's triplets and the mask
   extent before any curvature work
   (`inversion_interferometer_numba_util.symmetric_work_fraction_from`), at a cost of about 1 ms.
3. **Cost model.** The symmetric kernel takes `κ·f` of the full kernel's time, with **κ = 1.32**.
   κ was fitted on the synthetic geometry and checked out of sample on real Aris data at 1.48–1.53×
   predicted against 1.49–1.58× measured. So it wins whenever `κf < 1`, i.e. **f < 0.76**.
4. **Gate.** `interferometer_direct_conv_symmetric_max_work_fraction`:
   - The packaged default is **0.7**. At that value the predicted gain is at least 1.08×.
   - `0` always uses the full kernel.
   - `null` or unset means the default.

   The human chose the gate on 2026-10-16 (PyAutoArray#635).
5. **Ordering.** Columns are ranked by **ascending `ymin`**, the top extent row of each source
   column (stable sort, empty columns last). This gives each column the smallest possible row window.
6. **Routing.** The FFT-vs-`direct_conv` cost model (PyAutoArray#632) multiplies `t_dc` by
   `min(1, κf)` when the symmetric kernel will run, so the factory compares the kernel that will
   actually run.
7. **Expected sizes:**
   - Compact source columns (one lensed image per source pixel, small extent spans): f ≈ 0.40–0.42,
     about **1.8–1.9×** on the curvature step.
   - **Multiply imaged sources** (tall columns): f ≈ 0.50–0.51, about **1.5×**.
   - Above f ≈ 0.7 the kernel is not used.
8. **Leave the setting unset** unless running an A/B test. Results are identical to round-off
   whichever kernel runs, so this is a pure speed setting.

## Why it works, and the tall-column insight

`curvature_direct_conv` convolves every source column over the whole extent rectangle and gathers
every entry of its row of `F`. The symmetric kernel removes work that cannot change the result:

- **Symmetry.** Column `sp` of rank `r` computes only `F[sp, c]` for `rank[c] ≥ r`, and a mirror pass
  fills the rest. Each pixel that `A[:, c]` touches lies in an extent row `≥ ymin(c)`, so column `sp`
  needs `u = W̃ A[:, sp]` only on rows `≥ min over rank(c) ≥ r of ymin(c)`. In `ymin` order that
  minimum is the column's own top row.
- **Mask x-range.** Each extent row convolves only its masked x-span.
- **Gather.** The ranked upper triangle is gathered by a branch-free ranked-CSC sweep. Each computed
  entry is accumulated in the same order as the full kernel's CSR sweep, so it is **bit-identical**.

How much this saves depends on the vertical extent of the source columns:

- **Synthetic Aris-512 geometry** (3 nearest neighbours of a regular vertex grid): the median column
  spans 15 of 341 extent rows, so f ≈ 0.41.
- **Real Aris data** (SPT2319-55, step-8 Delaunay mesh): the median column spans **90 of 341 rows**
  (p95 158, max 165).
  - Multiple imaging feeds one source vertex from two to four lensed images that are far apart on
    the sky, so its column covers rows from several images.
  - The tallest columns are not the 30 mask-edge vertices: 0 of the 30 tallest columns are edge
    vertices.
  - Under the original centroid ordering this gave f = **0.587**, above the original 0.5 gate, so
    Aris stayed on the full kernel.
  - Ranking by `ymin` lowers f to **0.511**.
  - The ordering barely matters for compact columns (0.416 → 0.414 synthetic), but it matters for
    tall ones.

**Rule of thumb:** the gain is set by how tall the source-plane pixels' image-plane footprints are.
Lens models that multiply image a large part of the source should expect about 1.5×, not about 1.9×.

## Measurements

Timings are of the curvature step `F` only: `t_curvature_s`, the `inv.curvature_matrix` evaluation
after the mapper and kernel index arrays are built. Compile time is excluded: each figure is the
minimum of two repeats with a warm numba cache, and the first repeat matched the second within 3.3%.
**Synthetic and real data are kept in separate tables.**

### Real Aris data, RAL (the witness)

- **Data and model:** SPT2319-55 (2018.1.00285.S), 5,516,930 visibilities; 512×512 circular mask
  (r = 1.0″, 91,507 pixels, extent 341²); regular-overlay Delaunay mesh plus 30 edge vertices;
  ConstantSplit; the runner's maximum-likelihood fixed lens.
- **Host:** `euclid-ral-compute-1`, partition `ral`, `--exclusive`, `NUMBA_NUM_THREADS=1` (and OMP,
  MKL and OpenBLAS also at 1). The load average was ≤ 1.0 while timing.
- **Software:** PyAutoArray `e59d5fa9` plus the v3 patch (the code of PR #636 head `62a1e3ac`),
  PyAutoNerves `7f3055b`, PyAutoFit `9b394e73`, PyAutoGalaxy `9f9ac947`, PyAutoLens `4723668a`;
  numba 0.65.1, numpy 2.2.6, Python 3.12.
- **Jobs:**
  - 412490: step 8 and the synthetic witness.
  - 412491: the FFT arm at step 8.
  - 412492: steps 6 and 4.
  - 412493: the log-det control.

  Outputs are in `/mnt/ral/jnightin/scratch/aris_l2_20261016/`.

| Vertices (step) | nnz | f (ymin) | Full `direct_conv` | Symmetric | Speed-up | Predicted 1/(κf) | FFT route | Δ log L (rel) | Δ log det (abs) |
|---|---|---|---|---|---|---|---|---|---|
| 1,459 (8) | 273,753 | 0.511 | 22.43 s | 15.08 s | **1.49×** | 1.48× | 15.32 s | 0 (bit-identical) | 3.6e-11 |
| 2,577 (6) | 273,769 | 0.504 | 22.83 s | 14.95 s | **1.53×** | 1.50× | not run | 2.6e-16 | 8.3e-10 |
| 5,746 (4) | 273,909 | 0.495 | 24.22 s | 15.35 s | **1.58×** | 1.53× | not run | 2.2e-12 | 7.1e-3 |

- **Gate: passed.** The gate was ≥ 1.3× on the curvature step on real data, with a log likelihood
  matching to round-off. The whole figure of merit speeds up by 1.48× / 1.52× / 1.52×.
- **The 5,746-vertex log-det difference is round-off.**
  - The two F matrices agree to 4.4e-16 relative, which is the full kernel's own asymmetry.
  - cond(F+H) is 7.2e13.
  - Control job 412493 factorised the *full* kernel's own F twice, once from its lower triangle and
    once from its upper. The log det moved by **0.0077 nats**, the same size as the 0.0071 nats
    between the kernels.
- **Route at 1,459 vertices.**
  - Before this change the cost model sent 1,459 vertices to FFT.
  - On this node, FFT takes 15.32 s. The symmetric kernel takes 15.08 s, so the factory now picks
    `direct_conv` and is right, but only by **about 1.02×** over the previous default.
  - At 2,577 and 5,746 vertices, `direct_conv` was already the route, so the full 1.5–1.6× applies.
- **Calibration gap.** On RAL, FFT / full `direct_conv` is **0.68** at 1,459 vertices. The
  laptop-calibrated route constants predict 0.89. The FFT arm is relatively faster on this node than
  the #632 constants assume.

  This gap does not change any route measured here. It belongs to the next phase (FFT size levers),
  which should refit `c_fft / c_dc` on cluster nodes.
- The default route was checked live (inversion class) at every step: with the gate at 0.7 the
  factory picks `InversionInterferometerSparseNumba` with `use_symmetric_kernel=True`.
- **Memory (not part of the gate):** the witness process peaked at 43 GB RSS. It ran without NUFFT
  chunking and evaluated the 5.5M-visibility `fit.log_likelihood`. This peak was not attributed to
  a step, and the laptop run with `chunk_size=1_000_000` peaked at 10 GB. Neither kernel's own
  memory differs by more than its small index arrays.

### Real Aris data, laptop (the ordering check, before the RAL witness)

- **Setup:** the same data, mesh step 8, 1,459 vertices.
- **Host:** i9-10885H, WSL2, one thread, a shared machine under falling load (1.9–3.8 at the end).
  Read only the ratios.
- **Software:** PyAutoArray `e59d5fa9` plus the v2/v3 work-in-progress diff.

| Ordering | f | Full | Symmetric | Speed-up | Predicted 1/(κf) |
|---|---|---|---|---|---|
| Centroid row | 0.587 | 40.4 s | 30.5 s | 1.32× | 1.29× |
| `ymin` | 0.511 | 40.4 s | 27.6 s | 1.46× | 1.48× |

### Synthetic Aris-512 geometry, RAL

- **Geometry:** extent 341², circular mask with 91,361 pixels, vertices on a regular stride, 3-NN
  random weights, nnz 274,083.
- **Host and software:** the same as the real-data witness (job 412490, `ymin` order). The κ fit
  used the earlier centroid-order job 411835 on PyAutoArray `e59d5fa9` plus the v2 patch, on the
  same node.

| S | f (ymin) | Full (A, A) | Symmetric (C, C) | Speed-up | Predicted 1/(κf) | A-A / C-C spread |
|---|---|---|---|---|---|---|
| 1,428 | 0.414 | 22.27 / 22.08 s | 12.17 / 12.08 s | 1.83× | 1.83× | 1.008 / 1.007 |
| 2,540 | 0.408 | 22.53 / 22.55 s | 12.14 / 12.11 s | 1.86× | 1.86× | 1.001 / 1.002 |
| 5,712 | 0.403 | 23.96 / 23.94 s | 12.59 / 12.64 s | 1.90× | 1.88× | 1.001 / 1.004 |

The ranked triangle is bit-identical to the full kernel in all three. The full-matrix maximum
relative difference is ≤ 1.3e-15, which equals the full kernel's own asymmetry.

## Caveats

- **One dataset.** The real-data witness covers one dataset (Aris SPT2319-55), one mask (512×512)
  and one lens model. The tall-column result is a property of this lens configuration. Other
  multiply imaged systems should land near f ≈ 0.5, but this is not measured.
- **Serial only.** The parallel numba kernel was not given a symmetric variant. The campaign runs
  one thread per process (Nautilus parallelises over processes).
- **Fixed parameters.** κ is a single constant fitted at f ≈ 0.40–0.42 and checked at 0.50–0.51. It
  was not measured near the 0.7 gate. The predicted gain of 1.08× there is an extrapolation.
- **Not timed:** the 125×125 mask, where the 2026-10-15 research found the overhead cancels the gain
  (1.10× / 0.82×). There, f is about 0.6 with tiny absolute times. This was not re-timed with v3.

## Sources

- Library: PyAutoArray#635 (issue, with the 2026-10-16 checkpoint and the human decision),
  PyAutoArray#636 (PR).
- Campaign: PyAutoPulse#62, task `interferometer_large_mask_aris`, Knowledge (f) lever L2.
- Scripts (outside git, in the task scratch): `l2_realdata.py` (real-data A/B),
  `l2_realdata_route.py` (forced FFT arm), `l2_logdet_control.py` (triangle control),
  `l2_witness.py` (synthetic), `wf_orderings.py` (ordering and column-span statistics).
