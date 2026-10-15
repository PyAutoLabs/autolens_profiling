# Interferometer route gate, phase 2: the Aris routing table under the cost model (autolens_profiling#419)

Phase 2 of `interferometer-numba-route-gate`. In phase 1 (library), PyAutoArray#632 (issue
PyAutoArray#628, merge `17caf228`) replaced the fixed `interferometer_numba_nnz_per_source_max: 60`
gate with a two-constant cost model:

- `t_fft = c_fft · S · pocketfft-work(2ny × 2nx)`
- `t_dc = c_dc · nnz · ny · nx`

with `c_fft` 7.62e-10 and `c_dc` 1.77e-09. The model is in
`inversion_interferometer_numba_util.curvature_route_seconds_from`. The packaged default is now
`null` (cost model). A positive value is a hard gate, `0` turns numba off, and
`interferometer_route` forces a route.

This record re-measures the routing table on both Aris masks under the **default** settings. It
also carries the campaign's owed Aris log-det after-measurement from PyAutoArray#631 (issue
PyAutoArray#627). Campaign: PyAutoPulse#62, task `interferometer_large_mask_aris`.

## Verdict

- **The cost model routes all six cells to the faster measured arm.**
  - On the 125×125 mask (extent 83²) every cell goes to `direct_conv`, by 26–104× predicted.
  - On the 512×512 mask (extent 341²), 1,459 vertices goes to FFT, and 2,577 and 5,746 go to
    `direct_conv`.
  - At 2,577 the old gate of 60 sent the fit to FFT (nnz/col 106 > 60) although `direct_conv`
    measured 1.29× faster (34.2 vs 44.1 s/call). The cost model fixes that misroute.
- **The numba kill-gate question is answered by the cost model.** No fixed nnz/col value serves
  both masks:
  - The modelled crossover is ~309 nnz/col at extent 83² and ~166 at 341².
  - The measured crossover on the 512 mask lies between 106 and 188 nnz/col.
- **Log-det parity holds after #631.** numba and JAX `log_det_regularization_matrix_term` agree to
  ≤ 4.4e-8 absolute at all six cells.
  - At the investigation's original 125×125 vector, the pre-fix pair was numba 2443.22 vs JAX
    2561.98 (Δ −118.76).
  - It is now 2468.4463180 on both backends (Δ 3.9e-8).

## Six-cell table

All cells use the same parameter vector (below) and default settings. The predicted columns come
from `curvature_route_seconds_from(S, nnz, (ny, nx))` and are seconds on the calibration machine.
Only their ratio decides the route. The measured columns are **prior measurements on the laptop
(i9-10885H, single-thread numba), not re-timed**. They are steady medians from the Aris
investigation's PROGRESS3 (125, step7) and PROGRESS4 (512, step8) tables, on PyAutoArray
`ccddfba6` (before #630/#631/#632), and they are route-specific, not log-det-specific.

| Mask (extent) | Vertices | nnz/col | Chosen route (class) | Pred. t_fft (s) | Pred. t_dc (s) | Pred. t_fft/t_dc | Measured FFT s/call (NumPy) | Measured `direct_conv` s/call | JAX jit s/call | log-det numba | log-det JAX | Δ (numba − JAX) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 125×125 (83²) | 1,405 | 11.7 | `direct_conv` (`InversionInterferometerSparseNumba`) | 5.30 | 0.201 | 26.4 | not measured | 0.396 | 1.96 | 2468.4463180 | 2468.4463180 | 3.9e-8 |
| 125×125 (83²) | 2,947 | 5.57 | `direct_conv` (`InversionInterferometerSparseNumba`) | 11.11 | 0.200 | 55.5 | not measured | 2.08 | 6.84 | 4350.2973995 | 4350.2973995 | 2.1e-8 |
| 125×125 (83²) | 5,526 | 2.98 | `direct_conv` (`InversionInterferometerSparseNumba`) | 20.84 | 0.201 | 103.7 | not measured | 6.12 | 12.0 | 10157.1564483 | 10157.1564483 | 4.0e-8 |
| 512×512 (341²) | 1,459 | 187.6 | FFT (`InversionInterferometerSparse`) | 49.92 | 56.32 | 0.89 | 28.3 | 38.4 | 18.3 | 2573.1669411 | 2573.1669411 | 4.1e-8 |
| 512×512 (341²) | 2,577 | 106.2 | `direct_conv` (`InversionInterferometerSparseNumba`) | 88.18 | 56.33 | 1.57 | 44.1 | 34.2 | 32.2 | 4610.2972527 | 4610.2972526 | 2.6e-8 |
| 512×512 (341²) | 5,746 | 47.7 | `direct_conv` (`InversionInterferometerSparseNumba`) | 196.61 | 56.36 | 3.49 | 97.3 | 41.2 | not run (~72 extrapolated) | 10625.0105708 | 10625.0105708 | 4.4e-8 |

Reading the table:

- **Routes.** Measured FFT/`direct_conv` ratios on the 512 mask are 0.74 / 1.29 / 2.36. The
  predicted ratios are 0.89 / 1.57 / 3.49, so the faster arm is chosen in all three cells.
- **125 FFT arm.** The NumPy FFT arm was never timed on the 125 mask. The JAX jit column (the FFT
  route under JAX) is 2–5× slower than `direct_conv` there. The model predicts a far larger margin
  (26–104×) because the 166-point padded FFT is pocketfft's Bluestein case, which the model
  prices. Its absolute seconds do not transfer between clock states (the bake-off documents
  ~2.5× laptop drift).
- **The 125 numba 1,405 row** is the investigation's "his lens" re-run (step7.7). The 125 JAX
  rows use the same lens.
- **5,746 vertices on 512.** Before #630 the numba fit peaked at 7.17 GB RSS. In this run the
  whole script peaked at 2.49 GB: one numba fit plus a JAX inversion evaluated up to its
  regularization matrix only. That figure is an observation, not a #630 measurement.

## 125×125 original-vector log-det pair (PyAutoArray#627 / #631)

The vector is the first `numpy.random.default_rng(1)` draw over the 10-parameter model, at the
1,405-vertex mesh. It is the investigation's `vec0`.

| | numba (`xp=np`) | JAX (`xp=jnp`) | Δ |
|---|---|---|---|
| before (PyAutoArray `ccddfba6`) | 2443.22 | 2561.98 | −118.76 |
| after (PyAutoArray `17caf228`, post-#631) | 2468.4463180478 | 2468.4463180092 | 3.9e-8 |

The numba-route `figure_of_merit` at this vector is now 14094181.5649. Before the fix it was
14094168.9598 on numba and 14094228.3254 on JAX.

## Settings and provenance

- **Library.** PyAutoArray main `17caf228` (merge of #632; contains #630 `7cc71238` and #631
  `0377eeaa`). PyAutoGalaxy, PyAutoLens and PyAutoFit at the canonical mains of 2026-10-15.
  jax CPU, fp64.
- **Settings.** `al.Settings(use_positive_only_solver=False)` with
  `interferometer_numba_nnz_per_source_max` left at its packaged `None` (cost model) and
  `interferometer_route` at `None`. No explicit gate was set.
- **Model.** Delaunay with `AdaptSplit` regularization (`zeroed_pixels=30`). Image-plane meshes:
  - 125: regular `[::s, ::s]` grids, step 2 → 1,405 and step 1 → 5,526, plus the source-plane
    mesh → 2,947.
  - 512: steps 8 / 6 / 4 → 1,459 / 2,577 / 5,746.
  - Each mesh has 30 circle-edge points.
  - Isothermal + ExternalShear lens.
- **Data.** The Aris investigation's cached sparse datasets: 125×125 with extent 83², and
  512×512 at 3.0"/512 with extent 341² and 5.5M visibilities.
- **Fixed vector** (all six cells): `[-0.09511, -0.10142, -0.06813, -0.09916, 0.42328, -0.07500, -0.02795, 0.36657, 1.73001, 0.31102]`.
- **How each cell was measured.** Each cell ran in its own process: one eager numba-backend fit
  (route, nnz, S and log-det), then the JAX-backend inversion's
  `log_det_regularization_matrix_term`. That term is evaluated eagerly with `xp=jnp` and only up
  to the regularization matrix, which avoids the eager-JAX curvature memory at 512. Eager and jit
  agree to 1e-8 on this model (investigation step7.4b).
- **Host.** Laptop (i9-10885H, WSL2, shared; load 1.4–8.3 during the runs), 2026-10-15. Peak RSS
  per cell 1.65–3.04 GB. A 9 GB RSS watchdog was armed and never fired.
- The raw JSON and logs are kept in the session scratch and not committed. The table above is
  the record.

## Links

- PyAutoArray#628 / #632: cost-model route gate (`17caf228`).
- PyAutoArray#627 / #631: AdaptSplit log-det NumPy/JAX parity (`0377eeaa`).
- PyAutoArray#626 / #630: sparse data vector without the dense mapping matrix (`7cc71238`).
- PyAutoPulse#62: campaign "interferometer large-mask likelihood — memory and routing (Aris
  512×512)", task `interferometer_large_mask_aris`.
- Format and earlier crossover: [numba_interferometer_verdict.md](numba_interferometer_verdict.md)
  (#226) and [interferometer_mesh_cpu_breakdown_2026_09.md](interferometer_mesh_cpu_breakdown_2026_09.md).
- The three scripts that set `interferometer_numba_nnz_per_source_max` explicitly keep their
  hard-gate meaning and are unchanged.
