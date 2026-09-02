# Numpy CPU deflection angles — the *before* baseline (2026-09-02)

Phase 1, step 0 of the **`numpy-deflections-cpu`** epic
([PyAutoArray#514](https://github.com/PyAutoLabs/PyAutoArray/issues/514)): the measurement
package and its baseline artifacts, committed **before any library change**. Every number below
is the libraries exactly as they stand on `main` today.

Scripts: `scripts/lens/deflections/{total,dark,stellar}.py` (method and pin policy in
[`scripts/lens/deflections/README.md`](../../scripts/lens/deflections/README.md)).
Artifacts: `results/lens/deflections/<cell>_summary_<instrument>_v2026.8.17.1.{json,png}`.

## Environment

| | |
|---|---|
| Host | laptop CPU (WSL2), `OMP_NUM_THREADS=1`, numpy backend (`use_jax=False`) |
| PyAutoLens | v2026.8.17.1 |
| PyAutoArray | `302d5df3` |
| PyAutoGalaxy | `2ee44d50` |
| PyAutoLens (SHA) | `af514d179` |
| Grids | circular 3.5" mask; `over_sample_size_via_radial_bins_from(sub_size_list=[4, 2, 1], radial_list=[0.3, 0.6])`; `over_sample_size_pixelization=1` — identical to `scripts/imaging/likelihood_breakdown/pixelization_numba.py` |
| Repeats | 1 warm-up + median of 20 timed calls per (profile, grid); a separate 5-call cProfile pass for attribution only |

## The before table

Three timings per profile: the direct call on `dataset.grids.pixelization` (a `Grid2D`), the same
call on `dataset.grids.lp.over_sampled` (a `Grid2DIrregular`), and the deflection through a
two-plane `Tracer.traced_grid_2d_list_from`. "Non-finite" is the pinned count of non-finite
entries in the `Grid2D` deflection field.


### `hst` — 0.05"/px · 15361 Grid2D points · 17980 over-sampled points

| Profile | Cell | Grid2D s/call | Irregular s/call | Tracer s/call | Tracer / Grid2D | Non-finite |
|---------|------|---------------|------------------|---------------|-----------------|------------|
| `Isothermal` | total | 2.03 ms | 2.15 ms | 6.13 ms | 3.01× | 0 |
| `IsothermalSph` | total | 699.23 ms | 1.25 ms | 3.61 ms | 0.01× | 0 |
| `PowerLaw` | total | 11.17 ms | 12.66 ms | 25.10 ms | 2.25× | 0 |
| `PowerLawSph` | total | 701.00 ms | 1.53 ms | 3.66 ms | 0.01× | 2 |
| `NFW` | dark | 5.75 ms | 5.86 ms | 7.10 ms | 1.23× | 0 |
| `NFWSph` | dark | 590.89 ms | 9.91 ms | 14.23 ms | 0.02× | 0 |
| `gNFW` | dark | 357.66 ms | 445.91 ms | 721.18 ms | 2.02× | 0 |
| `gNFWSph` | dark | 1.205 s | 384.09 ms | 674.15 ms | 0.56× | 0 |
| `Gaussian` | stellar | 10.98 ms | 12.42 ms | 23.69 ms | 2.16× | 0 |
| `Gaussian_sph_case` | stellar | 13.00 ms | 21.95 ms | 53.38 ms | 4.11× | 0 |


### `euclid` — 0.1"/px · 3841 Grid2D points · 4468 over-sampled points

| Profile | Cell | Grid2D s/call | Irregular s/call | Tracer s/call | Tracer / Grid2D | Non-finite |
|---------|------|---------------|------------------|---------------|-----------------|------------|
| `Isothermal` | total | 1.02 ms | 1.54 ms | 1.93 ms | 1.89× | 0 |
| `IsothermalSph` | total | 179.78 ms | 298 µs | 1.03 ms | 0.01× | 0 |
| `PowerLaw` | total | 4.58 ms | 4.97 ms | 11.63 ms | 2.54× | 0 |
| `PowerLawSph` | total | 293.80 ms | 528 µs | 1.63 ms | 0.01× | 2 |
| `NFW` | dark | 1.06 ms | 1.10 ms | 2.85 ms | 2.69× | 0 |
| `NFWSph` | dark | 261.33 ms | 1.29 ms | 3.13 ms | 0.01× | 0 |
| `gNFW` | dark | 103.83 ms | 188.07 ms | 164.82 ms | 1.59× | 0 |
| `gNFWSph` | dark | 375.87 ms | 134.93 ms | 287.79 ms | 0.77× | 0 |
| `Gaussian` | stellar | 2.18 ms | 2.38 ms | 9.52 ms | 4.36× | 0 |
| `Gaussian_sph_case` | stellar | 7.08 ms | 6.35 ms | 10.09 ms | 1.43× | 0 |



## What the baseline says

### 1. Two profiles are expensive math: `gNFW` and `gNFWSph`

Both cost hundreds of milliseconds on **either** grid (hst: 358 ms / 446 ms and 1.20 s / 384 ms).
The generalised NFW has no closed-form deflection and is evaluated through a multi-Gaussian
expansion. cProfile charges 314 ms of `gNFW`'s 316 ms to
`mge.py:170(deflections_2d_via_mge_from)`, of which 308 ms is `mge.py:295(zeta_from)` and **283 ms
is 2 calls to the hand-rolled Faddeeva kernel `mge.py:79(wofz)`** in
`autogalaxy/profiles/mass/abstract/mge.py`. That is real vectorised work, it follows the profile
through the tracer too (`gNFW` tracer 721 ms on hst), and any speed-up has to come from the
expansion or the `wofz` kernel itself — not from the grid plumbing that dominates finding 2.

### 2. Every `*Sph` profile is orders of magnitude slower through `Grid2D` than `Grid2DIrregular`

On hst, `IsothermalSph` costs **699 ms** on `grids.pixelization` and **1.2 ms** on
`grids.lp.over_sampled` — a ~560× gap, in the wrong direction, since the irregular grid is the
*larger* of the two (17 980 vs 15 361 points). `PowerLawSph` (701 ms vs 1.5 ms) and `NFWSph`
(591 ms vs 9.9 ms) behave the same way. The elliptical siblings show no such gap at all
(`Isothermal` 2.0 ms vs 2.1 ms).

The cProfile chain below names the mechanism exactly. For `IsothermalSph`, of 1427 ms of
profiler-inflated call time:

    to_grid.py:11(via_grid_2d)                                    1425.94 ms
      builtins.getattr                                     x17    1425.62 ms
        uniform_2d.py:210(over_sampled)                     x1    1413.44 ms
          over_sample_util.py:377(grid_2d_slim_over_sampled_via_mask_from)
                                                            x1    1411.01 ms
            numpy linspace                               x30722    507.16 ms
            numpy meshgrid                               x15361    328.39 ms

The `Grid2D` branch of the `@to_grid` decorator
(`PyAutoArray/autoarray/structures/decorators/to_grid.py`) rebuilds a `Grid2D` from the result and
reads `getattr(result, "over_sampled", None)` off it. The spherical profiles return an
already-decorated `Grid2D` from their inner call, so that attribute access triggers the **lazy
over-sampled-grid build** — and that build is a per-pixel Python loop: `linspace` is called
30 722 times (2 x 15 361 masked pixels) and `meshgrid` 15 361 times, once per pixel. The elliptical
profiles return a raw ndarray, the `getattr` yields `None`, and nothing is built.

That per-pixel loop, not the lensing math, is the whole of the `*Sph` cost.

### 3. …but production ray-tracing does not pay it

The third timing is what stops (2) being over-read. `Tracer` receives the *same* `Grid2D` and is
fast — `IsothermalSph` 3.6 ms on hst, a `tracer/raw` of 0.01×. The reason is explicit in
`PyAutoLens/autolens/lens/tracer_util.py:traced_grid_2d_list_from`, which wraps the grid in
`aa.Grid2DIrregular(values=scaled_grid)` *before* calling `deflections_yx_2d_from`. So the penalty
sits on the **direct `Grid2D` API entry point** — what a user script, a plotter or a
convergence/potential helper reaches for — not on the likelihood's ray-trace. Anyone sizing the
win from (2) has to size it against that.

For the elliptical profiles the tracer ratio is the expected 1.2–3.0×: the same deflection, plus
grid bookkeeping and plane assembly.

### 4. `PowerLawSph` returns 2 non-finite deflection values

On both hst and euclid, exactly **2** of the `Grid2D` deflection entries are non-finite. It is the
only profile in the set that does. The count is pinned in its own right (`n_non_finite`) so a later
change to it — in either direction — is visible; `abs_sum` / `abs_max` are nan-aware reductions so
one bad point cannot collapse the whole pin to `NaN`.

## Measurement variance — read this before comparing an "after"

The **pinned values are exact**: all six cells were run three times on 2026-09-02 and every pinned
scalar and 16-coordinate sample reproduced within rtol 1e-6 (runs 2 and 3 both report
`Pinned-value checks PASSED`). The **timings are not** that stable. Median-of-20 per-call times
moved by up to ~±30% between the three passes on this host (laptop, WSL2) — e.g. `gNFWSph` on hst
came out at 877 ms, 1.04 s and 1.20 s; `Gaussian` on hst at 7.9 ms, 8.6 ms and 11.0 ms. Background
load on the machine is the cause, not the code.

So:

- **A later "after" must clear that band to count.** A 20% improvement measured once is inside the
  noise of this baseline. Take the win seriously at ~2× or better, or re-run both sides several
  times on a quiet machine.
- **The ratios are stable and are the real result.** The `Grid2D`-vs-`Grid2DIrregular` gap for the
  spherical profiles (two to three orders of magnitude) and the elliptical `tracer/raw` ratio
  (~2×) reproduced in all three passes. Findings 1–4 above rest on those, not on any third digit.

## cProfile attribution (hst, top 10 library frames by cumulative time per call)

Profiler-inflated numbers — they say *which* function the time is charged to, never how long it
takes. The headline timings in the table above were measured with the profiler off. The stored
`cprofile_top` in each JSON holds 12 rows; the harness's own closure frame (`_driver.py`) is
dropped here so every row below is library attribution.


**`IsothermalSph`** (699.23 ms per call)

| cumtime/call | tottime/call | calls | function |
|---|---|---|---|
| 1427.39 ms | 0.005 ms | 1 | `to_vector_yx.py:44(wrapper)` |
| 1427.38 ms | 0.012 ms | 4 | `abstract.py:95(result)` |
| 1427.22 ms | 0.023 ms | 4 | `abstract.py:91(evaluate_func)` |
| 1427.21 ms | 0.011 ms | 1 | `transform.py:63(wrapper)` |
| 1427.14 ms | 0.012 ms | 3 | `to_grid.py:53(wrapper)` |
| 1425.94 ms | 0.023 ms | 3 | `to_grid.py:11(via_grid_2d)` |
| 1425.62 ms | 0.017 ms | 17 | `~:0(<built-in method builtins.getattr>)` |
| 1413.44 ms | 2.367 ms | 1 | `uniform_2d.py:210(over_sampled)` |
| 1411.01 ms | 360.677 ms | 1 | `over_sample_util.py:377(grid_2d_slim_over_sampled_via_mask_from)` |
| 507.16 ms | 356.300 ms | 30722 | `function_base.py:25(linspace)` |

**`PowerLawSph`** (701.00 ms per call)

| cumtime/call | tottime/call | calls | function |
|---|---|---|---|
| 1116.34 ms | 0.003 ms | 1 | `to_vector_yx.py:44(wrapper)` |
| 1116.34 ms | 0.012 ms | 5 | `abstract.py:95(result)` |
| 1116.22 ms | 0.017 ms | 5 | `abstract.py:91(evaluate_func)` |
| 1116.21 ms | 0.008 ms | 1 | `transform.py:63(wrapper)` |
| 1115.62 ms | 0.009 ms | 3 | `to_grid.py:53(wrapper)` |
| 1114.73 ms | 0.019 ms | 3 | `to_grid.py:11(via_grid_2d)` |
| 1114.52 ms | 0.015 ms | 18 | `~:0(<built-in method builtins.getattr>)` |
| 1104.99 ms | 1.314 ms | 1 | `uniform_2d.py:210(over_sampled)` |
| 1103.63 ms | 281.734 ms | 1 | `over_sample_util.py:377(grid_2d_slim_over_sampled_via_mask_from)` |
| 395.76 ms | 277.595 ms | 30722 | `function_base.py:25(linspace)` |

**`NFWSph`** (590.89 ms per call)

| cumtime/call | tottime/call | calls | function |
|---|---|---|---|
| 1230.60 ms | 0.003 ms | 1 | `nfw.py:358(deflections_yx_2d_from)` |
| 1230.60 ms | 0.003 ms | 1 | `to_vector_yx.py:44(wrapper)` |
| 1230.59 ms | 0.013 ms | 5 | `abstract.py:95(result)` |
| 1230.47 ms | 0.020 ms | 5 | `abstract.py:91(evaluate_func)` |
| 1230.46 ms | 0.009 ms | 1 | `transform.py:63(wrapper)` |
| 1227.25 ms | 0.010 ms | 3 | `to_grid.py:53(wrapper)` |
| 1226.33 ms | 0.017 ms | 3 | `to_grid.py:11(via_grid_2d)` |
| 1226.08 ms | 0.016 ms | 18 | `~:0(<built-in method builtins.getattr>)` |
| 1216.43 ms | 1.418 ms | 1 | `uniform_2d.py:210(over_sampled)` |
| 1214.96 ms | 304.917 ms | 1 | `over_sample_util.py:377(grid_2d_slim_over_sampled_via_mask_from)` |

**`gNFW`** (357.66 ms per call)

| cumtime/call | tottime/call | calls | function |
|---|---|---|---|
| 315.95 ms | 0.006 ms | 1 | `gnfw.py:43(deflections_yx_2d_from)` |
| 315.95 ms | 0.029 ms | 1 | `gnfw.py:46(deflections_2d_via_mge_from)` |
| 315.88 ms | 0.007 ms | 1 | `to_vector_yx.py:44(wrapper)` |
| 315.87 ms | 0.009 ms | 3 | `abstract.py:95(result)` |
| 315.66 ms | 0.017 ms | 3 | `abstract.py:91(evaluate_func)` |
| 315.65 ms | 0.011 ms | 1 | `transform.py:63(wrapper)` |
| 314.14 ms | 4.240 ms | 1 | `mge.py:170(deflections_2d_via_mge_from)` |
| 308.00 ms | 23.624 ms | 1 | `mge.py:295(zeta_from)` |
| 284.35 ms | 283.037 ms | 2 | `mge.py:79(wofz)` |
| 1.80 ms | 0.011 ms | 2 | `to_grid.py:53(wrapper)` |

**`gNFWSph`** (1.205 s per call)

| cumtime/call | tottime/call | calls | function |
|---|---|---|---|
| 2468.68 ms | 0.008 ms | 1 | `gnfw.py:43(deflections_yx_2d_from)` |
| 2468.67 ms | 0.034 ms | 1 | `gnfw.py:46(deflections_2d_via_mge_from)` |
| 2468.56 ms | 0.010 ms | 1 | `to_vector_yx.py:44(wrapper)` |
| 2468.55 ms | 0.016 ms | 4 | `abstract.py:95(result)` |
| 2468.29 ms | 0.041 ms | 4 | `abstract.py:91(evaluate_func)` |
| 2468.27 ms | 0.020 ms | 1 | `transform.py:63(wrapper)` |
| 1917.59 ms | 0.020 ms | 3 | `to_grid.py:53(wrapper)` |
| 1916.90 ms | 0.022 ms | 2 | `to_grid.py:11(via_grid_2d)` |
| 1916.55 ms | 0.027 ms | 19 | `~:0(<built-in method builtins.getattr>)` |
| 1901.83 ms | 2.471 ms | 1 | `uniform_2d.py:210(over_sampled)` |


## Epic ledger — `numpy-deflections-cpu`

| | |
|---|---|
| Issue | [PyAutoArray#514](https://github.com/PyAutoLabs/PyAutoArray/issues/514) |
| Phase | 1 — measurement package + baseline |
| Opened | 2026-09-02 |
| Branch | `feature/numpy-deflections-p1` |

| Step | What | Status |
|------|------|--------|
| 0 | Measurement package `scripts/lens/deflections/` + BASELINE artifacts, committed **before** any library change | ✓ this commit (2026-09-02) |
| 1 | Diagnose the `Grid2D`-entry `over_sampled` cost and the `gNFW` MGE expansion cost | open |
| 2 | Library change(s) in PyAutoArray / PyAutoGalaxy | open |
| 3 | Re-run these cells; the pins must hold, or a `--repin` must carry a written reason | open |

**Rule for every later phase.** These cells are the after-measurement too. Re-run them with
`OMP_NUM_THREADS=1` on the same host, regenerate the dashboard
(`python scripts/misc/tooling/build_readme.py`), and read the pin column first: a timing
improvement alongside a `DRIFT` row is not an improvement, it is a changed answer.
