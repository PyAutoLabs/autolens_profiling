# Interferometer W~ curvature on the A100: pruned padded FFT and real-space pixel scale

PyAutoPulse task `interferometer_w_tilde_fft_size_levers` (campaign
[PyAutoPulse#62](https://github.com/PyAutoLabs/PyAutoPulse/issues/62)), issue
[#429](https://github.com/PyAutoLabs/autolens_profiling/issues/429). Follow-up to #320, lever 2
([interferometer_mesh_a100_breakdown_2026_09.md](interferometer_mesh_a100_breakdown_2026_09.md),
"Ranked levers" §2). Measurement only: no library change in this phase.

Witness JSON: [`results/breakdown/interferometer/jvla/delaunay_hpc_a100_fp64_fft_prune.json`](../breakdown/interferometer/jvla/delaunay_hpc_a100_fp64_fft_prune.json).
Script: `scripts/interferometer/delaunay/fft_size_levers.py`; submit
`hpc/batch_gpu/submit_fft_size_levers_interferometer_a100_fp64`.

## Verdict

1. **The pruned padded FFT passes both bars and clears the library-change threshold.** At jvla it
   takes 351 ms against the library's 500 ms for F, a saving of **29.7 %** of F (the threshold was
   10 %). The pruned F agrees with the library F to **1.7e-15** relative (bar 1e-12), and the figure
   of merit is unchanged to the last bit (bar 1e-6 nats). The fused F + solve + log-det + figure of
   merit pipeline is **1.41×** faster at jvla. The saving grows with the extent: 19.4 % at alma, 22.1 %
   at alma_high.

   Per the task's rule this files a PyAutoArray change. The human set no PyAutoArray change for this
   phase (2026-10-17), so the change is drafted as a prompt, not implemented (see "Next").
2. **A coarser real-space pixel scale does not hold the 0.5-nat bar.** At jvla the log evidence moves
   by **+1,310 / +1,807 / +2,757 nats** at 0.015 / 0.02 / 0.025″ against 0.01″. F does get cheaper
   (1.53× / 4.09× / 7.61×), but no coarser scale is inside the bar, so no workspace guidance prompt
   is filed.

## Setup

- **Host:** `euclid-ral-gpu-1`, partition `gpu`, `--gres=gpu:1 --exclusive`, one NVIDIA A100 80GB
  PCIe. Driver 610.57.04, CUDA UMD 13.3. Job **412515**, 21 min 08 s, `COMPLETED 0:0`.
- **Software:** JAX / jaxlib 0.10.2, nufftax 0.6.1, numpy 2.2.6, fp64 (`JAX_ENABLE_X64`).
  - Libraries: a private stack of the **2026.10.17.1** release tags, unpacked from `git archive`
    onto the node's local disk: PyAutoNerves `7f3055b7`, PyAutoFit `4b183e22`, PyAutoArray
    `e59d5fa9`, PyAutoGalaxy `35100def`, PyAutoLens `c90eceb0`. Not the shared RAL mirror.
  - autolens_profiling at this branch, `166fb9c1`.
  - The archive has no `.git`, so `revisions` reads "unavailable" in the JSON. The tags and SHAs are
    in `stack_revisions`.
- **XLA:** the flags the stack sets (`--xla_disable_hlo_passes=constant_folding
  --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false`) applied to both arms. The
  persistent compilation cache was warm (`cache_fresh: false`), so the recorded compile times
  (0.14–0.23 s for F) are cache loads. They are reported separately and are not part of any
  per-call figure.
- **Cell:** the #320 Delaunay cell. Hilbert-1500 image mesh on the lensed-source adapt image,
  `AdaptSplit(0.1, 10, 0.1)`, Isothermal + ExternalShear at the prior medians, sparse operator
  built with `method="nufft"`, batch size 128 (12 column blocks), PDIP solver.
- **Timing:** each F is jitted on its own. Lower, compile, first call and 10 steady calls are timed
  apart. Both are then re-timed as steady medians in A-B-B-A order (30 calls each after 5 warm-up
  calls), and the medians below are the mean of the two rounds. The rounds agree to ≤ 0.3 %.

## Pruned padded FFT (prune section)

The library pads each `(B, y, x)` column block to `(B, 2y, 2x)` and runs a full `rfft2` /
`irfft2`. The pruned apply is separable:

1. `rfft` along x with `n = 2x`, on the `y` non-zero rows only;
2. `fft` along y with `n = 2y`;
3. multiply by `Khat`;
4. `ifft` along y, keeping `y` rows;
5. `irfft` along x on those rows, keeping `x` columns.

Both x-stages then run on half the rows.

| Instrument | Extent (FFT grid) | Visibilities | F library | F pruned | F saving | Fused library | Fused pruned | Fused speed-up | F max rel diff | Δ figure of merit |
|---|---|---|---|---|---|---|---|---|---|---|
| alma | 140² (280²) | 1,000,000 | 17.97 ms | 14.49 ms | 19.4 % | 43.3 ms | 40.1 ms | 1.08× | 6.8e-16 | 0 |
| alma_high | 280² (560²) | 5,000,000 | 65.45 ms | 50.96 ms | 22.1 % | 89.5 ms | 76.0 ms | 1.18× | 1.0e-15 | 0 |
| jvla | 700² (1400²) | 25,000,000 | 499.6 ms | 351.2 ms | **29.7 %** | 528.5 ms | 375.2 ms | **1.41×** | 1.7e-15 | 0 |

- **"Fused"** is one jit of F → positive-only solve → the two log-dets → figure of merit, from the
  same triplets, data vector and regularization matrix.
- **The library F matches #320.** It reproduces 17.8 / 67.0 / 510.4 ms. Its standalone figure of
  merit matches the library fit to 1.5e-8 / 7.7e-7 / 1.4e-5 nats. Those are fp64 summation-order
  differences on figures of merit of order 1e7–3e8 nats.
- **The saving is above the Pulse task's estimate.** The task bounded it at ≤ 15–20 % of F, because
  it assumed the y-stages would dominate. The measured saving is 19–30 %, and it grows with the
  extent. At jvla it removes 148 ms of the 500 ms F, which is 28 % of the whole fused call.
- **The predicted no-go did not happen.** The task warned that one cuFFT 2-D call might beat four
  1-D calls. At every extent measured, the 1-D calls were faster.

## Real-space pixel scale at jvla (pixel_scale section)

- The jvla dataset (25M visibilities) was re-masked at each scale. The mask radius stayed at 3.5″
  and the field at 8″, so the grid is `round(8 / scale)` pixels on a side, rounded up to even.
- Each row's adapt image is the truth tracer's lensed source on that scale's grid, and the
  Hilbert-1500 mesh is rebuilt on it. The 0.01″ reference row is built the same way.
- "Full" is the jitted library `FitInterferometer.figure_of_merit`, compile timed apart (2.7–3.0 s).

| Scale | Grid | Extent (FFT grid) | Masked px | F | Full call | F speed-up | Log evidence | Shift vs 0.01″ |
|---|---|---|---|---|---|---|---|---|
| 0.01″ | 800² | 700² (1400²) | 384,852 | 499.5 ms | 538.6 ms | 1.00× | −301,296,978.51 | reference |
| 0.015″ | 534² | 466² (932²) | 171,004 | 326.8 ms | 358.5 ms | 1.53× | −301,295,668.19 | **+1,310.3** |
| 0.02″ | 400² | 350² (700²) | 96,224 | 122.2 ms | 150.3 ms | 4.09× | −301,295,171.03 | **+1,807.5** |
| 0.025″ | 320² | 280² (560²) | 61,572 | 65.6 ms | 93.2 ms | 7.61× | −301,294,221.06 | **+2,757.4** |

- **The 0.02″ prediction held.** The Pulse task predicted F at 0.02″ would land near the alma_high
  row (~67 ms). It is 122 ms (4.1×, against ~4× predicted from M). The 0.025″ row, with the same
  280² extent as alma_high, matches alma_high's F (65.6 vs 65.5 ms).
- **0.015″ gains only 1.53×, though M falls 2.25×.** The padded FFT length 932 = 2²·233 has a
  large prime factor, so cuFFT pays the awkward-length penalty the CPU cost model already prices
  (PyAutoArray#632).
- **Which terms move.** Most of the shift is in the log-determinant terms:
  - `log det(F + H)`: 20,741 → 18,448 / 16,929 / 15,844.
  - χ² gets slightly *worse* as the scale coarsens: +263 / +71 / +335.
  - The regularization term grows: 1.9 → 10.9 / 13.9 / 62.3.

  So the evidence rises because the coarser image grid constrains the 1,500 source pixels less, not
  because the fit improves.
- **The shift has two causes.** The mesh is rebuilt on each scale's adapt image, so it confounds
  the pixel scale with a change of mesh. Either way, the evidence moves by thousands of nats across
  these scales, about 2,600–5,500 times the 0.5-nat bar.
- **Only the evidence was compared.** Reconstructions on different meshes are not directly
  comparable, so none was compared.

## Not measured (out of scope or closed)

- **Rectangular mesh.** The Pulse task lists Delaunay *and* rect. Only Delaunay was run. The F
  build does not depend on the mesh beyond S and nnz, and both are fixed here (S = 1,500).
- **The NumPy branch.** The NumPy branch of `apply_operator` (`scipy.fft`) would get the same
  pruning, but it was not measured. The human closed the interferometer CPU lane on 2026-10-17.
- **The FFT route's cost-model constants on RAL CPU.** These were planned as part A of #429 and
  dropped when the CPU lane closed. The evidence in hand stays open; see the campaign journal of
  2026-10-17. The constants refit is deferred.

## Next

- **A PyAutoArray prompt (drafted, not filed into development).** Make
  `InterferometerSparseOperator.apply_operator` compute the separable pruned transform on the JAX
  branch. Expected gain on the A100: 19–30 % of F and 1.08–1.41× on the fused call, with F
  unchanged to ~1e-15. Any NumPy-branch change waits for the CPU lane to reopen.
- **No workspace docs prompt for a coarser real-space grid:** the bar failed.

## Sources

- RAL job 412515. SLURM log and JSON are in `/mnt/ral/jnightin/scratch/fft_levers_jobs/` and
  `/mnt/ral/jnightin/scratch/fft_levers_results/`.
- The Pulse task `interferometer_w_tilde_fft_size_levers` (original prompt, filed 2026-09-27).
- The #320 A100 breakdown JSONs: `results/breakdown/interferometer/{alma,alma_high,jvla}/delaunay_hpc_a100_fp64.json`
  (alma at `results/breakdown/interferometer/delaunay_hpc_a100_fp64.json`).
