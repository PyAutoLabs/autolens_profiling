"""
Numpy CPU Profiling: Deflection Angles — Stellar Mass Profiles
==============================================================

Times ``deflections_yx_2d_from`` for the elliptical and spherical-case Gaussian on the grids a real
imaging fit uses, with **no library change applied** — this cell is the *before*
measurement for the ``numpy-deflections-cpu`` epic (PyAutoArray#514, phase 1).

The Gaussian is the atom of an MGE mass basis: a 60-component MGE lens pays
this cost sixty times per likelihood evaluation, so a millisecond here is a
tenth of a second there. ``autogalaxy.mp`` has no ``GaussianSph`` — the
spherical case is the elliptical ``Gaussian`` with ``ell_comps=(0.0, 0.0)``,
recorded as ``Gaussian_sph_case`` so the two rows stay distinguishable.

Three timings per profile, all of them calls into the installed library:

1. ``Grid2D`` — ``dataset.grids.pixelization`` (sub-size 1), the ray-trace grid.
2. ``Grid2DIrregular`` — ``dataset.grids.lp.over_sampled``, the over-sampled
   light-profile grid.
3. ``Tracer`` — the same deflection through a two-plane
   ``traced_grid_2d_list_from``, so the wrapper cost is measured, not assumed.

Method, pin policy and the ``--repin`` contract all live in ``_driver.py``;
the fiducial parameters live in ``_profiles.py``. This file is deliberately
thin: a prologue, the pinned expectations, and one call.

Output
------
``results/lens/deflections/stellar_summary_<instrument>_v<version>.{json,png}``
"""

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)

_sys.path.insert(0, str(_profiling_root()))

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os  # noqa: E402
import sys as _smoke_sys  # noqa: E402

import _driver  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

# The pinned deflection values for this cell, keyed by instrument then profile:
# ``abs_sum`` / ``abs_max`` over the full Grid2D deflection field, and a
# 16-coordinate (y, x) ``sample`` at the fixed arcsec positions in
# ``_driver.PIN_COORDINATES``. Checked at rtol 1e-6 on every run. Filled
# automatically on the first run for an instrument; moved only via
# ``--repin --repin-reason "..."``. Do not hand-edit.
# >>> BEGIN EXPECTED (generated — rewrite with --repin) >>>
EXPECTED: dict[str, dict[str, dict]] = {
    "euclid": {
        "Gaussian": {
            "abs_sum": 4180.333936988291,
            "abs_max": 1.004598819684994,
            "n_non_finite": 0,
            "sample": [
                [-0.08522849052042059, 0.8223777592115288],
                [0.5347232749072048, 0.5347232749072048],
                [0.8223777592115288, -0.08522849052042047],
                [0.6240102400561088, -0.6240102400561087],
                [0.08522849052042097, -0.8223777592115287],
                [-0.5347232749072052, -0.5347232749072051],
                [-0.8223777592115287, 0.08522849052042086],
                [-0.6240102400561088, 0.6240102400561087],
                [-0.016565763250107057, 0.14931860950686004],
                [0.14931860950686004, -0.016565763250107043],
                [0.016565763250108057, -0.1493186095068597],
                [-0.1493186095068597, 0.01656576325010803],
                [0.6179244621552085, 0.6179244621552086],
                [0.5837883073848549, -0.5837883073848548],
                [-0.6179244621552097, -0.6179244621552095],
                [-0.5837883073848549, 0.5837883073848548],
            ],
        },
        "Gaussian_sph_case": {
            "abs_sum": 3590.885170005575,
            "abs_max": 0.9024533744335073,
            "n_non_finite": 0,
            "sample": [
                [0.0, 0.7869386805747332],
                [0.5564496774123882, 0.5564496774123883],
                [0.7869386805747332, 4.818609681455442e-17],
                [0.5564496774123883, -0.5564496774123882],
                [9.637219362910884e-17, -0.7869386805747332],
                [-0.5564496774123883, -0.5564496774123882],
                [-0.7869386805747332, 4.818609681455442e-17],
                [-0.5564496774123882, 0.5564496774123883],
                [0.0, 0.14915940518355933],
                [0.14915940518355933, 9.133379406038452e-18],
                [1.8266758812076904e-17, -0.14915940518355933],
                [-0.14915940518355933, 9.133379406038452e-18],
                [0.4950550061047418, 0.49505500610474185],
                [0.49505500610474185, -0.4950550061047418],
                [-0.49505500610474185, -0.4950550061047418],
                [-0.4950550061047418, 0.49505500610474185],
            ],
        },
    },
    "hst": {
        "Gaussian": {
            "abs_sum": 16710.89642173353,
            "abs_max": 1.004598819684994,
            "n_non_finite": 0,
            "sample": [
                [-0.08522849052042059, 0.8223777592115288],
                [0.5347232749072048, 0.5347232749072048],
                [0.8223777592115288, -0.08522849052042047],
                [0.6240102400561088, -0.6240102400561087],
                [0.08522849052042097, -0.8223777592115287],
                [-0.5347232749072052, -0.5347232749072051],
                [-0.8223777592115287, 0.08522849052042086],
                [-0.6240102400561088, 0.6240102400561087],
                [-0.016565763250107057, 0.14931860950686004],
                [0.14931860950686004, -0.016565763250107043],
                [0.016565763250108057, -0.1493186095068597],
                [-0.1493186095068597, 0.01656576325010803],
                [0.6179244621552085, 0.6179244621552086],
                [0.5837883073848549, -0.5837883073848548],
                [-0.6179244621552097, -0.6179244621552095],
                [-0.5837883073848549, 0.5837883073848548],
            ],
        },
        "Gaussian_sph_case": {
            "abs_sum": 14359.300577425658,
            "abs_max": 0.9024533744335073,
            "n_non_finite": 0,
            "sample": [
                [0.0, 0.7869386805747332],
                [0.5564496774123882, 0.5564496774123883],
                [0.7869386805747332, 4.818609681455442e-17],
                [0.5564496774123883, -0.5564496774123882],
                [9.637219362910884e-17, -0.7869386805747332],
                [-0.5564496774123883, -0.5564496774123882],
                [-0.7869386805747332, 4.818609681455442e-17],
                [-0.5564496774123882, 0.5564496774123883],
                [0.0, 0.14915940518355933],
                [0.14915940518355933, 9.133379406038452e-18],
                [1.8266758812076904e-17, -0.14915940518355933],
                [-0.14915940518355933, 9.133379406038452e-18],
                [0.4950550061047418, 0.49505500610474185],
                [0.49505500610474185, -0.4950550061047418],
                [-0.49505500610474185, -0.4950550061047418],
                [-0.4950550061047418, 0.49505500610474185],
            ],
        },
    },
}
# <<< END EXPECTED <<<

_driver.run(
    cell="stellar",
    profiles=["Gaussian", "Gaussian_sph_case"],
    cell_path=_Path(__file__).resolve(),
    expected=EXPECTED,
)
