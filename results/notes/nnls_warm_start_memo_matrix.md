# NNLS warm-start memo: model robustness matrix

Aggregated from `results/breakdown/imaging/delaunay_numba_nnls_iterations_*_v2026.8.17.1.json`
by `scripts/imaging/likelihood_breakdown/nnls_iterations_matrix.py`. Companion to
[`nnls_warm_start_memo.md`](./nnls_warm_start_memo.md), which measured the single fiducial.

Each model variant changes exactly ONE thing about that fiducial (Delaunay Hilbert-1250 +
ConstantSplit + MGE-60 linear lens light + Isothermal + shear), memo OFF then ON on the
*same* seeded instance sequences: `rw` = random walk (N(0, 1% of each prior's width) per
step, the sampler-like regime), `iid` = independent draws from the central 20% of every
prior (uncorrelated, pessimistic). Laptop CPU fp64, `OMP_NUM_THREADS=1`,
`AUTOARRAY_NUMBA_OPERATED_MEMO=0`. `ratio` = median iterations OFF / ON (>1 = memo saved
iterations). `err frac` = warm-start errors / solve size — the quantity a fallback
tolerance would threshold on. `Δlnl` = max relative log-likelihood deviation OFF vs ON.

Three variants could not be a pure one-thing change, and say so here rather than in a
footnote: `rectangular` must drop `ConstantSplit` for `Constant` (split regularization is
structurally incompatible with the rectangular interpolator); `source_complex` ships as
data/noise/PSF with no truth tracer, so its adapt image is the positive-clipped data, and
it is sampled at 0.05"/px, i.e. hst geometry; `no_edge_zeroing` flips the solve onto the
full-system branch but does NOT change the index set — `Delaunay(zeroed_pixels=0)` already
keeps every index, and the only mesh here that zeroes any (`rectangular`, its shape-derived
136-pixel border) zeroes a *static* set. So the memo key never churns in this matrix.

| instrument | model | seq | n | iters med OFF→ON | iters max OFF→ON | errs med OFF→ON | errs max ON | solve s med OFF→ON | solve s max ON | eval s med OFF→ON | ratio | err frac ON med/p90/max | err frac OFF med | Δlnl | retries | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| euclid | fiducial | rw | 20 | 70→7 | 206→84 | 74→9 | 77 | 0.257→0.054 | 0.353 | 0.73→0.51 | 10.00× | 0.007/0.011/0.059 | 0.056 | 1.1e-14 | 0 | 0 |
| euclid | fiducial | iid | 20 | 86→94 | 134→146 | 96→96 | 139 | 0.392→0.269 | 0.495 | 0.99→0.78 | 0.91× | 0.073/0.098/0.106 | 0.073 | 8.6e-15 | 0 | 0 |
| euclid | powerlaw | rw | 20 | 60→8 | 72→72 | 69→10 | 75 | 0.281→0.051 | 0.354 | 0.87→0.56 | 7.93× | 0.007/0.021/0.057 | 0.053 | 2.6e-15 | 0 | – |
| euclid | powerlaw | iid | 20 | 92→85 | 129→117 | 108→96 | 122 | 0.426→0.278 | 0.482 | 0.99→0.81 | 1.09× | 0.073/0.088/0.093 | 0.082 | 7.8e-15 | 0 | – |
| euclid | subhalo | rw | 20 | 76→6 | 88→73 | 81→10 | 73 | 0.404→0.071 | 0.383 | 1.05→0.73 | 11.69× | 0.008/0.016/0.056 | 0.062 | 1.5e-14 | 0 | – |
| euclid | subhalo | iid | 20 | 88→94 | 123→140 | 98→95 | 128 | 0.580→0.370 | 0.610 | 1.41→0.98 | 0.93× | 0.073/0.096/0.098 | 0.074 | 6.4e-15 | 0 | – |
| euclid | no_lens_light | rw | 20 | 59→14 | 67→59 | 96→18 | 97 | 0.289→0.066 | 0.348 | 0.56→0.32 | 4.07× | 0.014/0.026/0.078 | 0.077 | 4.5e-16 | 0 | – |
| euclid | no_lens_light | iid | 20 | 53→101 | 69→160 | 91→117 | 161 | 0.300→0.306 | 0.580 | 0.56→0.58 | 0.52× | 0.094/0.120/0.129 | 0.073 | 2.2e-16 | 0 | – |
| euclid | sersic_light | rw | 20 | 42→5 | 48→37 | 154→6 | 154 | 0.315→0.053 | 0.284 | 0.63→0.33 | 8.50× | 0.005/0.013/0.123 | 0.123 | 3.1e-15 | 0 | – |
| euclid | sersic_light | iid | 20 | 44→41 | 64→63 | 157→62 | 152 | 0.369→0.223 | 0.441 | 0.69→0.65 | 1.06× | 0.049/0.068/0.122 | 0.125 | 4.1e-15 | 0 | – |
| euclid | rectangular | rw | 20 | 598→7 | 630→602 | 507→11 | 501 | 0.727→0.035 | 0.980 | 1.37→0.71 | 85.50× | 0.010/0.025/0.436 | 0.441 | 1.2e-15 | 0 | – |
| euclid | rectangular | iid | 20 | 620→161 | 677→619 | 502→158 | 519 | 0.558→0.233 | 0.480 | 1.13→0.85 | 3.85× | 0.138/0.158/0.452 | 0.436 | 6.7e-15 | 0 | – |
| euclid | adapt_reg | rw | 20 | 66→16 | 86→72 | 71→18 | 66 | 0.512→0.111 | 0.836 | 1.32→0.70 | 4.26× | 0.013/0.022/0.050 | 0.054 | 2.4e-14 | 0 | – |
| euclid | adapt_reg | iid | 20 | 70→70 | 99→165 | 75→65 | 105 | 0.352→0.270 | 0.637 | 1.00→0.85 | 1.00× | 0.050/0.067/0.080 | 0.057 | 4.3e-14 | 0 | – |
| euclid | no_edge_zeroing | rw | 20 | 70→7 | 206→84 | 74→9 | 77 | 0.359→0.057 | 0.593 | 0.99→0.63 | 10.00× | 0.007/0.011/0.059 | 0.056 | 1.1e-14 | 0 | – |
| euclid | no_edge_zeroing | iid | 20 | 86→94 | 134→146 | 96→96 | 139 | 0.523→0.324 | 0.825 | 1.26→0.87 | 0.91× | 0.073/0.098/0.106 | 0.073 | 8.6e-15 | 0 | – |
| euclid | mesh_600 | rw | 20 | 39→6 | 47→38 | 43→8 | 43 | 0.064→0.010 | 0.074 | 0.50→0.43 | 6.50× | 0.013/0.022/0.065 | 0.065 | 7.6e-16 | 0 | – |
| euclid | mesh_600 | iid | 20 | 49→56 | 66→79 | 54→54 | 74 | 0.065→0.050 | 0.122 | 0.50→0.51 | 0.88× | 0.082/0.106/0.112 | 0.082 | 5.3e-15 | 0 | – |
| euclid | mesh_2000 | rw | 20 | 96→13 | 134→85 | 117→17 | 110 | 1.376→0.304 | 1.347 | 2.61→1.30 | 7.38× | 0.008/0.013/0.053 | 0.057 | 7.7e-16 | 0 | – |
| euclid | mesh_2000 | iid | 20 | 114→128 | 179→200 | 132→133 | 184 | 1.601→1.208 | 1.899 | 3.10→2.29 | 0.88× | 0.065/0.083/0.089 | 0.064 | 1.3e-14 | 0 | – |
| hst | fiducial | rw | 20 | 32→8 | 41→41 | 31→10 | 31 | 0.205→0.063 | 0.217 | 1.61→1.39 | 4.27× | 0.007/0.014/0.024 | 0.024 | 1.3e-15 | 0 | 0 |
| hst | fiducial | iid | 20 | 43→70 | 71→99 | 48→66 | 102 | 0.241→0.237 | 0.387 | 1.59→1.43 | 0.62× | 0.051/0.064/0.078 | 0.037 | 9.5e-15 | 0 | 3 |
| hst | powerlaw | rw | 20 | 29→9 | 37→28 | 29→13 | 31 | 0.248→0.088 | 0.346 | 1.98→1.97 | 3.22× | 0.010/0.016/0.024 | 0.022 | 1.7e-15 | 0 | – |
| hst | powerlaw | iid | 20 | 47→62 | 69→115 | 48→66 | 90 | 0.357→0.301 | 0.552 | 2.30→2.03 | 0.75× | 0.050/0.063/0.069 | 0.037 | 2.7e-14 | 0 | – |
| hst | rectangular | rw | 20 | 886→8 | 906→881 | 802→13 | 783 | 1.212→0.040 | 0.905 | 3.17→1.89 | 118.07× | 0.011/0.021/0.681 | 0.698 | 2.9e-15 | 0 | – |
| hst | rectangular | iid | 20 | 872→172 | 993→854 | 763→99 | 732 | 1.090→0.327 | 1.307 | 3.10→2.03 | 5.09× | 0.086/0.111/0.637 | 0.664 | 9.9e-15 | 0 | – |
| hst | no_edge_zeroing | rw | 20 | 32→8 | 41→41 | 31→10 | 31 | 0.284→0.084 | 0.270 | 2.24→1.81 | 4.27× | 0.007/0.014/0.024 | 0.024 | 1.3e-15 | 0 | – |
| hst | no_edge_zeroing | iid | 20 | 43→73 | 71→99 | 48→68 | 102 | 0.342→0.341 | 0.515 | 2.16→2.32 | 0.59× | 0.052/0.064/0.078 | 0.037 | 1.1e-14 | 0 | – |
| hst | source_complex | rw | 20 | 322→95 | 361→311 | 492→94 | 478 | 0.384→0.090 | 0.404 | 1.25→0.87 | 3.39× | 0.072/0.108/0.365 | 0.375 | 4.1e-16 | 0 | 0 |
| hst | source_complex | iid | 20 | 344→475 | 427→605 | 514→393 | 526 | 0.356→0.372 | 0.531 | 1.16→1.21 | 0.73× | 0.300/0.343/0.402 | 0.393 | 2.2e-16 | 0 | 0 |
| hst | mesh_600 | rw | 20 | 24→3 | 29→21 | 21→4 | 20 | 0.059→0.012 | 0.048 | 1.84→1.91 | 8.00× | 0.005/0.017/0.030 | 0.032 | 1.0e-15 | 0 | – |
| hst | mesh_600 | iid | 20 | 30→36 | 51→54 | 34→32 | 47 | 0.069→0.041 | 0.064 | 1.88→1.79 | 0.85× | 0.048/0.067/0.071 | 0.052 | 6.4e-15 | 0 | – |

## Findings

- **Worst cell by iteration ratio:** `euclid/no_lens_light/iid` at 0.52× (53 → 101 median iterations).
- **Worst cell by MAX memo-ON iterations:** `hst/rectangular/rw` at 881 (memo OFF max on the same sequence: 906), and its worst evaluation is index 0. In 18/32 cells the memo-ON maximum IS evaluation 0, whose memo is empty and which therefore ran the dense-sign start — a cold start, not a pathology. The cells where it is not: `euclid/fiducial/iid` (idx 12), `euclid/powerlaw/iid` (idx 15), `euclid/subhalo/iid` (idx 13), `euclid/no_lens_light/iid` (idx 12), `euclid/sersic_light/iid` (idx 19), `euclid/adapt_reg/iid` (idx 3), `euclid/no_edge_zeroing/iid` (idx 12), `euclid/mesh_600/iid` (idx 12), `euclid/mesh_2000/iid` (idx 12), `hst/fiducial/iid` (idx 2), `hst/powerlaw/iid` (idx 15), `hst/no_edge_zeroing/iid` (idx 2), `hst/source_complex/iid` (idx 2), `hst/mesh_600/iid` (idx 16).
- **Cells where median memo-ON solve seconds EXCEED memo-OFF (2/32):** `hst/source_complex/iid` (0.356 → 0.372 s), `euclid/no_lens_light/iid` (0.300 → 0.306 s).
- **Retries** (a memo-seeded solve raised and the library fell back to the dense-sign start): none in any cell.
- **Tolerance fallbacks** (a memo seed's error fraction exceeded `nnls_warm_start_error_tolerance` × its entry's dense-sign reference, so the entry was dropped): `hst/fiducial/iid` (3) — of 6/32 cells recorded with the guard.
- **Parity:** no cell exceeds 1e-08; worst is `euclid/adapt_reg/iid` at max rel Δlnl 4.3e-14, max |Δreconstruction| 8.1e-10.

### Fallback-tolerance calibration (memo-seed error fraction)

- **random_walk** (16 cells): memo-ON error fraction median 0.005–0.072 (across-cell median 0.008), per-cell max up to 0.681; dense-sign (OFF) median 0.022–0.698.
- **iid** (16 cells): memo-ON error fraction median 0.048–0.300 (across-cell median 0.073), per-cell max up to 0.637; dense-sign (OFF) median 0.037–0.664.

- **Separating X.** Helpful cells (ratio ≥ 1×): 21; non-helpful (< 1×): 11. The two populations OVERLAP: helpful cells reach a median memo-ON error fraction of 0.138 while the least-wrong non-helpful cell sits at 0.048. **No X separates them**, so a fallback tolerance cannot be calibrated from the error fraction alone. The highest cut that never discards a seed which was still saving iterations is X ≈ 0.14, but it fires on none of the non-helpful cells below it, i.e. the fallback would be inert where it is wanted.

- **Relative seed quality (ON / OFF median error fraction).** The absolute fraction cannot separate the populations, but the ratio nearly does: helpful cells top out at 0.89, non-helpful cells start at 0.76, with the worst seed in the matrix at 1.42 (`hst/no_edge_zeroing/iid`). A fallback phrased as *"the memo seed is no better than the dense-sign start"* — i.e. **ON/OFF error fraction ≳ 0.9** — is therefore the discriminator the raw fraction is not. Its cost is that computing the OFF fraction means computing the dense-sign start, which is the work the memo exists to skip.

- **Observability caveat.** The error count is only known once the solve has run, so a fallback cannot threshold on the current evaluation's fraction. Any X would have to be applied to the *previous* evaluation's fraction as a running proxy — which is exactly the quantity the random-walk/iid contrast above shows is stable within a sequence and unstable across regimes.
