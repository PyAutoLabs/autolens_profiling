# Numpy CPU deflection angles — phase 1 before/after (2026-09-02)

> Part 1 below is the **before** baseline, measured before any library change.
> Part 2 ("After phase 1") is the same cells re-run against the three library fixes.

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

---

# After phase 1 — the library change measured (2026-09-02)

Same host, same day, same cells, same pins as the before pass above. The three library
commits under test:

| Repo | SHA | What it does |
|---|---|---|
| PyAutoArray | `92bb9b2c` | `to_grid` reads `_over_sampled` / `_over_sampler` instead of the public property, and `Grid2D.over_sampled` short-circuits to `self` at sub-size 1 |
| PyAutoGalaxy | `2605c924` | `Galaxy` skips the second deflection call for the over-sampled grid at sub-size 1 |
| PyAutoLens | `b1515e5d3` | `Tracer` skips the second ray-trace of the over-sampled grid at sub-size 1 |

Parents are exactly the before-pass SHAs (`302d5df3` / `2ee44d50` / `af514d179`), so this is a
one-commit-per-repo A/B.

**Every pin PASSED at rtol 1e-6 on all six cells. No `--repin` was used, `pinned_drift` is empty
everywhere, and `pin_provenance` is `null`.** The answers are bit-identical; only the cost moved.

## The after table

### `hst` — 0.05"/px · 15361 Grid2D points · 17980 over-sampled points

| Profile | Cell | Grid2D before → after | Grid2D × | Irregular before → after | Irreg × | Tracer before → after | Tracer × | Non-finite | Pin |
|---------|------|-----------------------|----------|--------------------------|---------|-----------------------|----------|------------|-----|
| `Isothermal` | total | 2.03 ms → 2.07 ms | **0.98×** | 2.15 ms → 2.21 ms | 0.97× | 6.13 ms → 2.70 ms | **2.27×** | 0 | PASS |
| `IsothermalSph` | total | 699.23 ms → 985 µs | **709.95×** | 1.25 ms → 932 µs | 1.34× | 3.61 ms → 1.43 ms | **2.52×** | 0 | PASS |
| `PowerLaw` | total | 11.17 ms → 9.14 ms | **1.22×** | 12.66 ms → 10.04 ms | 1.26× | 25.10 ms → 9.00 ms | **2.79×** | 0 | PASS |
| `PowerLawSph` | total | 701.00 ms → 1.23 ms | **569.89×** | 1.53 ms → 1.21 ms | 1.26× | 3.66 ms → 1.68 ms | **2.17×** | 2 | PASS |
| `NFW` | dark | 5.75 ms → 3.53 ms | **1.63×** | 5.86 ms → 3.85 ms | 1.52× | 7.10 ms → 3.81 ms | **1.86×** | 0 | PASS |
| `NFWSph` | dark | 590.89 ms → 4.17 ms | **141.78×** | 9.91 ms → 4.34 ms | 2.28× | 14.23 ms → 4.35 ms | **3.27×** | 0 | PASS |
| `gNFW` | dark | 357.66 ms → 292.78 ms | **1.22×** | 445.91 ms → 343.15 ms | 1.30× | 721.18 ms → 280.71 ms | **2.57×** | 0 | PASS |
| `gNFWSph` | dark | 1.205 s → 300.94 ms | **4.00×** | 384.09 ms → 336.95 ms | 1.14× | 674.15 ms → 275.06 ms | **2.45×** | 0 | PASS |
| `Gaussian` | stellar | 10.98 ms → 11.07 ms | **0.99×** | 12.42 ms → 13.70 ms | 0.91× | 23.69 ms → 13.96 ms | **1.70×** | 0 | PASS |
| `Gaussian_sph_case` | stellar | 13.00 ms → 12.86 ms | **1.01×** | 21.95 ms → 14.12 ms | 1.55× | 53.38 ms → 13.62 ms | **3.92×** | 0 | PASS |

### `euclid` — 0.1"/px · 3841 Grid2D points · 4468 over-sampled points

| Profile | Cell | Grid2D before → after | Grid2D × | Irregular before → after | Irreg × | Tracer before → after | Tracer × | Non-finite | Pin |
|---------|------|-----------------------|----------|--------------------------|---------|-----------------------|----------|------------|-----|
| `Isothermal` | total | 1.02 ms → 711 µs | **1.44×** | 1.54 ms → 735 µs | 2.10× | 1.93 ms → 934 µs | **2.07×** | 0 | PASS |
| `IsothermalSph` | total | 179.78 ms → 382 µs | **470.63×** | 298 µs → 279 µs | 1.07× | 1.03 ms → 577 µs | **1.78×** | 0 | PASS |
| `PowerLaw` | total | 4.58 ms → 2.59 ms | **1.77×** | 4.97 ms → 2.85 ms | 1.74× | 11.63 ms → 3.01 ms | **3.87×** | 0 | PASS |
| `PowerLawSph` | total | 293.80 ms → 510 µs | **576.24×** | 528 µs → 411 µs | 1.29× | 1.63 ms → 679 µs | **2.41×** | 2 | PASS |
| `NFW` | dark | 1.06 ms → 961 µs | **1.10×** | 1.10 ms → 975 µs | 1.13× | 2.85 ms → 1.19 ms | **2.40×** | 0 | PASS |
| `NFWSph` | dark | 261.33 ms → 1.14 ms | **228.60×** | 1.29 ms → 1.11 ms | 1.16× | 3.13 ms → 1.31 ms | **2.38×** | 0 | PASS |
| `gNFW` | dark | 103.83 ms → 54.60 ms | **1.90×** | 188.07 ms → 75.84 ms | 2.48× | 164.82 ms → 71.04 ms | **2.32×** | 0 | PASS |
| `gNFWSph` | dark | 375.87 ms → 61.39 ms | **6.12×** | 134.93 ms → 81.71 ms | 1.65× | 287.79 ms → 93.22 ms | **3.09×** | 0 | PASS |
| `Gaussian` | stellar | 2.18 ms → 2.68 ms | **0.81×** | 2.38 ms → 2.98 ms | 0.80× | 9.52 ms → 3.04 ms | **3.13×** | 0 | PASS |
| `Gaussian_sph_case` | stellar | 7.08 ms → 2.93 ms | **2.42×** | 6.35 ms → 3.18 ms | 2.00× | 10.09 ms → 3.25 ms | **3.11×** | 0 | PASS |

## What moved, and why

Two independent mechanisms, and they land on two different columns.

**1. The `*Sph` penalty was on the direct-`Grid2D` entry point only — and it is gone.**
`IsothermalSph` 699 ms → 0.99 ms (710×), `PowerLawSph` 701 ms → 1.23 ms (570×), `NFWSph` 591 ms →
4.17 ms (142×) on hst; 229–576× on euclid. This is finding 2 of the before pass closed out. The
`@to_grid` decorator no longer reaches through the public `over_sampled` property when it rebuilds
a `Grid2D`, so the lazy per-pixel `linspace`/`meshgrid` build in
`over_sample_util.grid_2d_slim_over_sampled_via_mask_from` is never triggered — and where it *is*
legitimately reached at sub-size 1, `Grid2D.over_sampled` now returns `self` instead of building an
identical copy. The `*Sph` `Grid2D` column now sits alongside its `Grid2DIrregular` sibling, which
is what it should always have been.

`gNFWSph` is the partial case: 1.205 s → 301 ms (4.0×) on hst. The wrapper cost came off, but the
multi-Gaussian-expansion cost underneath it (finding 1) did not — that is real vectorised math in
`mge.py:zeta_from` → `wofz`, and it is untouched by this phase.

**2. The tracer double trace was the likelihood-side win.** Every profile's `Tracer` column
improves by ~1.7–3.9× (hst) and ~1.8–3.9× (euclid), including the ellipticals that had no `*Sph`
problem at all. `Tracer.traced_grid_2d_list_from` was ray-tracing the grid *and* its `over_sampled`
companion; on `dataset.grids.pixelization` (`over_sample_size_pixelization=1`) those two grids are
point-for-point identical, so exactly half the ray-tracing was duplicate work. Removing it gives
the flat ~2× that shows up across the whole table.

The elliptical `Grid2D` and `Irregular` columns are, correctly, roughly flat (0.8×–1.9×) — neither
mechanism touches them, and the spread there is the host variance described below, not a result.

## Likelihood breakdown — a control test, and a negative result

The four numba likelihood cells were run as a **paired A/B on this host** rather than against the
git-tracked `v2026.8.17.1` artifacts. Those tracked artifacts turned out not to be a valid before:
they were written by an earlier harness (commit `d8122b3`, a different step decomposition — the
`delaunay`/`euclid` cell now splits MGE and curvature into sub-steps the old file does not have) on
a quieter machine, and every unrelated step in them (Cholesky `log det`, BLAS solve, FFT convolve)
is 1.3–1.6× faster than this host manages today. Comparing against them would have reported host
drift as a code result.

The A/B instead ran each cell twice per arm, interleaved, with the *only* difference being
`PYTHONPATH` pointed at the three parent SHAs versus the three fixed SHAs. The control arm was
verified to reproduce the bug first (`IsothermalSph` `Grid2D` 985 ms on the parent SHAs versus
0.99 ms on the fixed ones), so the harness is demonstrably wired to the right code on both sides.

| Cell | `Inversion build` before → after | `direct_log_likelihood_function_per_call` before → after | Pin |
|---|---|---|---|
| `pixelization_numba` hst | 35.64 ms → 32.20 ms (1.11×) | 348.8 ms → 407.5 ms (0.86×) | PASS — 27661.910133664103 |
| `pixelization_numba` euclid | 23.87 ms → 25.67 ms (0.93×) | 146.2 ms → 153.8 ms (0.95×) | no pin defined; `log_likelihood` bit-identical |
| `delaunay_numba` hst | 44.23 ms → 39.57 ms (1.12×) | 594.7 ms → 813.3 ms (0.73×) | PASS — 29090.527192092646 |
| `delaunay_numba` euclid | 28.56 ms → 18.11 ms (1.58×) | 411.4 ms → 405.3 ms (1.02×) | PASS — 7215.3687893658935 |

**This is a negative result, and it is the honest one.** The ratios scatter 0.73×–1.58× with no
sign, i.e. the change is not measurable here. That is expected rather than disappointing: the
duplicate ray-trace this phase removes is worth ~1–2 ms inside a 20–45 ms `Inversion build` step,
which is well inside this host's variance. What the A/B *does* establish, on all four cells and
both arms, is that **`log_likelihood` is bit-identical between the parent and fixed SHAs** —
27661.91013366411, 6213.306873885871, 29090.527210448134, 7215.369093236959 — and every defined pin
PASSES on both arms.

Because these four runs are a load-confounded no-op, the tracked
`results/breakdown/imaging/*_v2026.8.17.1.json` artifacts were **left at their committed values**
rather than overwritten. Replacing a quiet-machine record with a loaded-machine one that measures
the same thing would regress the timing board for no information.

## `jax_compile` warm compile — inconclusive, no pin change

`scripts/misc/jax_compile/probe.py --model-type pixelization --instrument hst --transforms jit` is
the cell that traces a `Tracer` through `jax.jit`, so it is where a removed duplicate subgraph would
show. Run cold-then-warm on both arms against a private cache dir:

| Arm | trace | warm compile | 1-min load average |
|---|---|---|---|
| parent SHAs | 15.13 s | 321 ms | 7.51 |
| fixed SHAs | 4.80 s | 268 ms | 4.69 |

Warm compile came out 321 ms → 268 ms (1.20×), but the two arms ran at 7.5 and 4.7 load average and
this README's own measurement discipline records host load making compile timings wrong by up to
**7×**. So this is **not** a result: it is consistent with a small win and equally consistent with
noise. The standing pin for this cell (`imaging/pixelization/hst` · `jit` = 216 ms, 2026-07-28) was
**not** re-pinned and is not contradicted — both arms sit above it by the same host factor. The
probe's corpus writes were reverted so four load-contaminated records do not enter `pins.json`.

Re-run on a quiet host, or on RAL, if the compile axis is wanted as evidence.

## Workspace smoke

`autolens_workspace/scripts/imaging/features/pixelization/cpu_fast_modeling.py`, run from the
canonical `autolens_workspace` root against the worktree libraries under the standard smoke profile
(`PYAUTO_TEST_MODE=2`, `PYAUTO_SMALL_DATASETS=1`, `PYAUTO_DISABLE_JAX=1`, …):

**exit 0, 47.56 s wall.** (Note: this script is not itself in `autolens_workspace/smoke_tests.txt`
— only its `multi_galaxy/` sibling is — so it was run directly rather than through
`pyauto-heart smoke`.)

## Measurement variance — the same caveat still applies

The before pass recorded ±30% run-to-run movement in median-of-20 per-call timings on this laptop,
and this pass confirms it: the control arm measured `IsothermalSph` at 985 ms today against the
before pass's 699 ms for *identical* code, and one `stellar`/`euclid` cell had to be re-run after
`Gaussian` came out at 5.09 ms on a contended first pass versus 2.68 ms clean. A background
`remote-dev-serv` process held this host at load 4–7 for much of the session.

So read this table the way the before pass asked to be read:

- **The `*Sph` `Grid2D` result (142×–710×) and the `Tracer` result (~2–4× across every profile) are
  far outside that band and are real.** Both have a named mechanism and a reproduced control.
- **The 0.8×–1.9× movements in the elliptical `Grid2D` / `Irregular` columns are noise**, in both
  directions, and should not be quoted as either a win or a regression.
- **The breakdown and `jax_compile` rows are inside the band** and are reported as no-change.

## Epic ledger — `numpy-deflections-cpu`

| Step | What | Status |
|------|------|--------|
| 0 | Measurement package `scripts/lens/deflections/` + BASELINE artifacts | ✓ 2026-09-02 |
| 1 | Diagnose the `Grid2D`-entry `over_sampled` cost and the `gNFW` MGE expansion cost | ✓ 2026-09-02 |
| 2 | Library change(s) in PyAutoArray / PyAutoGalaxy / PyAutoLens | ✓ `92bb9b2c` / `2605c924` / `b1515e5d3` |
| 3 | Re-run these cells; pins must hold | ✓ this commit — all six cells PASS, no `--repin` |
| 4 | The `gNFW` / `gNFWSph` MGE expansion cost (finding 1) — still open | open |

Finding 1 of the before pass is the remaining target: `gNFW` still costs 293 ms on hst and
`gNFWSph` 301 ms, essentially all of it in `mge.py:wofz`. Phase 1 removed the wrapper around that
math; it did not make the math cheaper.

---

# After phase 2 — `scipy.special.wofz` + the exact spherical MGE branch (2026-09-02)

Phase 1 removed the *wrapper* around the MGE math and left finding 1 open: `gNFW` still cost
293 ms on hst and `gNFWSph` 301 ms, essentially all of it in the hand-rolled Faddeeva kernel
`mge.py:wofz`. Phase 2 is the library change that goes after that math, filed as
[PyAutoGalaxy#596](https://github.com/PyAutoLabs/PyAutoGalaxy/issues/596).

| Repo | SHA | What it does |
|---|---|---|
| PyAutoGalaxy | `09785e32` | `MGEDecomposer.wofz` dispatches to `scipy.special.wofz` on numpy (the hand-rolled rational approximation is kept for JAX tracing); `Gaussian.wofz` deduped onto it; `_wofz_masked` skips the second Faddeeva call where the Gaussian envelope has underflowed; new `_spherical_mge_deflections_from` takes the exact q → 1 radial form on the numpy path |

Parent is `e76c062e` (the phase-1 merge), so this is again a one-commit A/B, and the "before"
column below is the **committed phase-1 after** artifact for every cell.

**The JAX path is untouched and bit-identical** (a jitted `gNFW` deflection matches to max abs
diff 0.0): the dispatch only changes what runs when `xp is np`.

## Why the answers moved — the two mechanisms, adjudicated

Both mechanisms move pinned values, and in both cases the **new** value is the more accurate one.
Neither is a tolerance being loosened.

### 1. Faddeeva accuracy — the routine that was replaced is the inaccurate side

Against an **mpmath reference at dps 40**, over the argument range these profiles actually
evaluate:

| Faddeeva implementation | max relative error vs mpmath (dps 40) |
|---|---|
| hand-rolled `mge.py:wofz` rational approximation (the old numpy path) | **3.0e-6** |
| `scipy.special.wofz` (the new numpy path) | **1.3e-14** |

So the ~6-significant-figure routine is the one being retired. Every wofz-driven pin move below
is ≤ ~4e-6 relative — exactly the size of the old routine's own error — and it moves *towards*
the reference, not away from it.

The `_wofz_masked` skip is separately bounded: only `Im(z) >= 0` is ever passed, where
`|w(z)| <= 1`, so the term dropped where the Gaussian envelope has underflowed is negligible to
< 1e-15 — three orders of magnitude below float64 round-off on these sums, and nine below the
error it replaces.

### 2. The spherical branch — the clamp bias measured by lifting the clamp

The MGE deflection is an elliptical formula; the spherical members used to reach it with the axis
ratio **clamped to `q = 0.9999`**, because the elliptical form is singular at q = 1. The new
branch instead evaluates the exact q → 1 limit,

    alpha_r(r) = sum_j 2 A_j sigma_j^2 (1 - exp(-r^2 / 2 sigma_j^2)) / r

with no Faddeeva call at all. That it is the right limit was checked by **lifting the clamp** and
letting the elliptical path converge to it:

| q | `gNFWSph` rel. difference (elliptical path vs the exact form) | `Gaussian` (q → 1) |
|---|---|---|
| 0.999 | 6.4e-4 | 8.4e-4 |
| 0.99999 | 6.4e-6 | 8.4e-6 |
| 0.999999 | 6.4e-7 | 8.4e-7 |

Linear in (1 − q), to the exact form, in both profiles. Read back at the clamp itself
(1 − q = 1e-4) that is a **~6e-5 relative bias** the old spherical deflections carried — which is
precisely the size of the `abs_sum` / `abs_max` pin moves recorded below.

The clamp also produced **spurious cross-axis deflections**: a strictly spherical profile has no
tangential component, but at q = 0.9999 the elliptical formula returns a tiny non-zero one. The
16-coordinate sample pins caught two of them, and both are now **exactly 0** (`gNFWSph`
−3.078e-8 → 0; `Gaussian_sph_case` −6.593e-37 → 0). Those are the only two pin entries whose
*relative* shift is 1.0, which is why `--repin-force` was required.

## Pin drift — every moved value, with its mechanism

`total.py` (`Isothermal`, `IsothermalSph`, `PowerLaw`, `PowerLawSph`) has **no MGE profile**, and
its pins were re-run on both instruments and **PASSED untouched** — no `--repin`, `pinned_drift`
empty, `pin_provenance` `null`. The same is true of `NFW` and `NFWSph` inside `dark.py`: both have
closed-form deflections that never enter `mge.py`, and both diffed at exactly `0.000e+00` on every
pinned scalar and all 32 sample values. That is the control: only the MGE-routed profiles moved.

| Profile · pin | old (phase-1) | new (phase 2) | rel. shift | Mechanism |
|---|---|---|---|---|
| `gNFW.abs_sum` (hst) | 64242.42298664075 | 64242.42549321639 | 3.90e-08 | scipy wofz |
| `gNFW.abs_max` (hst) | 3.5464245674350785 | 3.5464245920500037 | 6.94e-09 | scipy wofz |
| `gNFW.sample[29]` (both) | −2.3386045451451705 | −2.3386138347416985 | **3.97e-06** | scipy wofz — inside the retired routine's own 3.0e-6 error |
| `gNFW.abs_sum` (euclid) | 16071.750238845096 | 16071.750916861929 | 4.22e-08 | scipy wofz |
| `gNFW.abs_max` (euclid) | 3.541525075621922 | 3.5415250998628403 | 6.85e-09 | scipy wofz |
| `gNFWSph.abs_sum` (hst) | 70483.143338306 | 70480.27024532984 | **4.08e-05** | q = 0.9999 clamp bias removed |
| `gNFWSph.abs_max` (hst) | 3.930477262532153 | 3.930229113965614 | **6.31e-05** | clamp bias removed |
| `gNFWSph.abs_sum` (euclid) | 17629.28040028201 | 17628.561740673365 | **4.08e-05** | clamp bias removed |
| `gNFWSph.abs_max` (euclid) | 3.9238007576720246 | 3.923553220872915 | **6.31e-05** | clamp bias removed |
| `gNFWSph.sample[0]` (both) | −3.078160248954665e-08 | **0.0** | 1.00e+00 | spurious cross-axis deflection → exactly 0 |
| `Gaussian.abs_sum` (hst) | 16710.89555761894 | 16710.89642173353 | 5.17e-08 | scipy wofz |
| `Gaussian.abs_max` (both) | 1.0045988177091016 | 1.004598819684994 | 1.97e-09 | scipy wofz |
| `Gaussian.sample[2]` (both) | 0.5347249918530897 | 0.5347232749072048 | **3.21e-06** | scipy wofz |
| `Gaussian.abs_sum` (euclid) | 4180.333690680411 | 4180.333936988291 | 5.89e-08 | scipy wofz |
| `Gaussian_sph_case.abs_sum` (hst) | 14360.33826028308 | 14359.300577425658 | **7.23e-05** | clamp bias removed |
| `Gaussian_sph_case.abs_max` (both) | 0.9025078735164493 | 0.9024533744335073 | **6.04e-05** | clamp bias removed |
| `Gaussian_sph_case.abs_sum` (euclid) | 3591.144756659599 | 3590.885170005575 | **7.23e-05** | clamp bias removed |
| `Gaussian_sph_case.sample[16]` (both) | −6.592519057047366e-37 | **0.0** | 1.00e+00 | spurious cross-axis deflection → exactly 0 |

`n_non_finite` is unchanged everywhere (0 for every MGE profile; `PowerLawSph`'s 2 is in the
untouched `total` cell).

`dark.py` and `stellar.py` were re-pinned on both instruments with
`--repin --repin-reason "…" --repin-force`, the diff read first in a refused (non-forced) pass;
`--repin-force` was needed **only** because of the two exact-zero entries, whose relative shift is
1.0 by construction. Each re-pinned cell was then re-run **without** `--repin`: all four
**PASSED** at rtol 1e-6. The reason string is stored as `pin_provenance` in the re-pin run's JSON.

## The after table — phase-1 after (committed) → phase 2

Same host, same day, `OMP_NUM_THREADS=1`, median of 20 timed calls. Only the four MGE-routed
profiles are expected to move; the rest are printed in the dashboard and are noise in both
directions.

### `hst` — 0.05"/px · 15361 Grid2D points · 17980 over-sampled points

| Profile | Cell | Grid2D before → after | Grid2D × | Irregular before → after | Irreg × | Tracer before → after | Tracer × | Pin |
|---------|------|-----------------------|----------|--------------------------|---------|-----------------------|----------|-----|
| `gNFW` | dark | 292.78 ms → 95.62 ms | **3.06×** | 343.15 ms → 117.62 ms | 2.92× | 280.71 ms → 99.62 ms | **2.82×** | re-pinned, PASS |
| `gNFWSph` | dark | 300.94 ms → 4.52 ms | **66.56×** | 336.95 ms → 5.11 ms | 65.96× | 275.06 ms → 4.79 ms | **57.39×** | re-pinned, PASS |
| `Gaussian` | stellar | 11.07 ms → 6.56 ms | **1.69×** | 13.70 ms → 8.31 ms | 1.65× | 13.96 ms → 7.94 ms | **1.76×** | re-pinned, PASS |
| `Gaussian_sph_case` | stellar | 12.86 ms → 2.13 ms | **6.05×** | 14.12 ms → 2.27 ms | 6.23× | 13.62 ms → 2.81 ms | **4.85×** | re-pinned, PASS |
| `NFW` | dark | 3.53 ms → 2.23 ms | 1.58× | 3.85 ms → 2.46 ms | 1.57× | 3.81 ms → 2.63 ms | 1.45× | PASS untouched |
| `NFWSph` | dark | 4.17 ms → 2.99 ms | 1.39× | 4.34 ms → 3.25 ms | 1.33× | 4.35 ms → 3.20 ms | 1.36× | PASS untouched |
| `Isothermal` | total | 2.07 ms → 1.99 ms | 1.04× | 2.21 ms → 2.08 ms | 1.07× | 2.70 ms → 2.41 ms | 1.12× | PASS untouched |
| `IsothermalSph` | total | 985 µs → 962 µs | 1.02× | 932 µs → 944 µs | 0.99× | 1.43 ms → 1.59 ms | 0.90× | PASS untouched |
| `PowerLaw` | total | 9.14 ms → 10.55 ms | 0.87× | 10.04 ms → 12.52 ms | 0.80× | 9.00 ms → 10.84 ms | 0.83× | PASS untouched |
| `PowerLawSph` | total | 1.23 ms → 1.39 ms | 0.89× | 1.21 ms → 1.39 ms | 0.87× | 1.68 ms → 1.86 ms | 0.91× | PASS untouched |

### `euclid` — 0.1"/px · 3841 Grid2D points · 4468 over-sampled points

| Profile | Cell | Grid2D before → after | Grid2D × | Irregular before → after | Irreg × | Tracer before → after | Tracer × | Pin |
|---------|------|-----------------------|----------|--------------------------|---------|-----------------------|----------|-----|
| `gNFW` | dark | 54.60 ms → 28.24 ms | **1.93×** | 75.84 ms → 33.40 ms | 2.27× | 71.04 ms → 29.75 ms | **2.39×** | re-pinned, PASS |
| `gNFWSph` | dark | 61.39 ms → 1.63 ms | **37.56×** | 81.71 ms → 1.70 ms | 48.03× | 93.22 ms → 1.83 ms | **51.00×** | re-pinned, PASS |
| `Gaussian` | stellar | 2.68 ms → 1.81 ms | **1.48×** | 2.98 ms → 1.91 ms | 1.56× | 3.04 ms → 2.05 ms | **1.48×** | re-pinned, PASS |
| `Gaussian_sph_case` | stellar | 2.93 ms → 546 µs | **5.36×** | 3.18 ms → 530 µs | 6.00× | 3.25 ms → 734 µs | **4.42×** | re-pinned, PASS |
| `NFW` | dark | 961 µs → 900 µs | 1.07× | 975 µs → 924 µs | 1.06× | 1.19 ms → 1.22 ms | 0.97× | PASS untouched |
| `NFWSph` | dark | 1.14 ms → 1.12 ms | 1.02× | 1.11 ms → 1.12 ms | 1.00× | 1.31 ms → 1.31 ms | 1.00× | PASS untouched |
| `Isothermal` | total | 711 µs → 696 µs | 1.02× | 735 µs → 657 µs | 1.12× | 934 µs → 900 µs | 1.04× | PASS untouched |
| `IsothermalSph` | total | 382 µs → 340 µs | 1.12× | 279 µs → 269 µs | 1.04× | 577 µs → 548 µs | 1.05× | PASS untouched |
| `PowerLaw` | total | 2.59 ms → 2.51 ms | 1.03× | 2.85 ms → 2.72 ms | 1.05× | 3.01 ms → 2.63 ms | 1.14× | PASS untouched |
| `PowerLawSph` | total | 510 µs → 463 µs | 1.10× | 411 µs → 362 µs | 1.13× | 679 µs → 622 µs | 1.09× | PASS untouched |

### How much of that is real

The same ±30% host band as the two earlier passes applies, and the non-MGE rows show it directly:
`PowerLaw` came out 0.87× on hst and 1.03× on euclid for **identical, untouched code**, and hst
`NFW` reads 1.58× against euclid's 1.07× for the same reason. Read those rows as noise.

The MGE rows were measured three times each on hst during the re-pin sequence, and the spread is
the honest error bar on the headline ratios:

| Profile (hst, Grid2D) | run 1 | run 2 | run 3 (committed) | before | ratio range |
|---|---|---|---|---|---|
| `gNFW` | 110.8 ms | 121.8 ms | 95.6 ms | 292.8 ms | 2.4× – 3.1× |
| `gNFWSph` | 5.20 ms | 5.18 ms | 4.52 ms | 300.9 ms | 58× – 67× |
| `Gaussian` | 6.11 ms | 6.04 ms | 6.56 ms | 11.1 ms | 1.7× – 1.8× |
| `Gaussian_sph_case` | 1.40 ms | 1.61 ms | 2.13 ms | 12.9 ms | 6× – 9× |

So: **`gNFWSph` ~40–67× and `Gaussian_sph_case` ~4–9× are the two large, unambiguous wins**
(the spherical branch removes the Faddeeva evaluation entirely, not just speeds it up), and
**`gNFW` ~2.4–3.1× and `Gaussian` ~1.5–1.8×** are the wofz-dispatch wins — smaller, still well
outside the band, and consistent across both instruments.

## Likelihood breakdown — pins hold, artifacts not overwritten

`scripts/imaging/likelihood_breakdown/pixelization_numba.py` was run on both instruments as a
correctness check: its fiducial mass profile is `Isothermal` (no MGE), and its MGE *light*
profiles go through a different code path that this change does not touch.

| Cell | Pin | Result |
|---|---|---|
| `pixelization_numba` hst | 27661.910133665442 | **PASSED** — `log_likelihood` 27661.91013366411 |
| `pixelization_numba` euclid | none defined | `log_likelihood` **6213.306873885871** — bit-identical to the phase-1 value |

As in phase 1, the tracked `results/breakdown/imaging/*_v2026.8.17.1.json` artifacts were
**reverted rather than committed**: these were load-contaminated re-runs of a quiet-host record and
replacing them would regress the timing board for no information. Only the pin verdicts above are
taken from them.

## Re-scoped targets

The phase-2 prompt asked for 5× on `gNFW`, 20× on `gNFWSph` and 1.5× on `Gaussian`. Measured
against the committed phase-1 baseline, the achieved ceilings are:

| Profile | Asked | Measured | Verdict |
|---|---|---|---|
| `gNFW` | 5× | ~2.5–3.1× | **short** — the Faddeeva call is still there, scipy just evaluates it faster and correctly |
| `gNFWSph` | 20× | ~40–67× | **exceeded** — the branch removes the Faddeeva evaluation outright |
| `Gaussian` (elliptical) | 1.5× | ~1.1–1.8× | **met, marginally** — same mechanism as `gNFW`, on a much smaller array |
| `Gaussian` (q = 1) | — | ~4–9× | spherical branch, as `gNFWSph` |

`gNFW` is the one that falls short, and the reason is structural rather than fixable by another
kernel swap: an elliptical MGE deflection *needs* the Faddeeva function, and `scipy.special.wofz`
is already a well-optimised C implementation. Getting to 5× would take reducing the number of
Gaussians in the expansion, or evaluating the (30, N) array in a lower precision where that is
defensible — both of which change the answer and belong in their own phase with their own
adjudication.

## Filed follow-up — the JAX path

The JAX path still runs the hand-rolled rational approximation, still clamps spherical profiles to
q = 0.9999, and is therefore **still carrying both defects this phase fixed on numpy** — the 3.0e-6
Faddeeva error and the ~6e-5 clamp bias, including the spurious cross-axis deflections. That was
deliberate here (keeping the JAX arithmetic bit-identical made this a clean single-axis change),
and it is filed for its own task:

- `PyAutoMind/draft/research/autogalaxy/jax_faddeeva_seams_and_spherical_clamp_audit.md`

## Epic ledger — `numpy-deflections-cpu`

| Step | What | Status |
|------|------|--------|
| 0 | Measurement package `scripts/lens/deflections/` + BASELINE artifacts | ✓ 2026-09-02 |
| 1 | Diagnose the `Grid2D`-entry `over_sampled` cost and the `gNFW` MGE expansion cost | ✓ 2026-09-02 |
| 2 | Phase 1 library change (PyAutoArray `92bb9b2c` / PyAutoGalaxy `2605c924` / PyAutoLens `b1515e5d3`) | ✓ 2026-09-02 |
| 3 | Phase 1 re-run; pins held, no `--repin` | ✓ 2026-09-02 |
| 4 | Phase 2 library change — scipy `wofz` + exact spherical MGE branch (PyAutoGalaxy `09785e32`, issue #596) | ✓ 2026-09-02 |
| 5 | Phase 2 re-run; `total` pins held, `dark` / `stellar` re-pinned with provenance | ✓ this commit |
| 6 | JAX-path audit — Faddeeva seams and the spherical clamp | filed (PyAutoMind draft) |
