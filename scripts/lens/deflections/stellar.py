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
            "abs_sum": 4180.333690680411,
            "abs_max": 1.0045988177091016,
            "n_non_finite": 0,
            "sample": [
                [-0.08522848494462226, 0.8223777453345469],
                [0.5347249918530897, 0.5347249214028542],
                [0.8223777453345469, -0.08522848494462215],
                [0.6240102336059056, -0.6240102336059056],
                [0.08522848494462248, -0.8223777453345469],
                [-0.5347249918530889, -0.534724921402854],
                [-0.8223777453345469, 0.08522848494462237],
                [-0.6240102336059056, 0.6240102336059056],
                [-0.0165657619170127, 0.14931861195098112],
                [0.14931861195098112, -0.016565761917012686],
                [0.016565761917012825, -0.149318611950981],
                [-0.149318611950981, 0.016565761917012797],
                [0.6179239070956012, 0.6179239070956013],
                [0.5837883046591865, -0.5837883046591864],
                [-0.6179239070956013, -0.6179239070956012],
                [-0.5837883046591865, 0.5837883046591864],
            ],
        },
        "Gaussian_sph_case": {
            "abs_sum": 3591.144756659599,
            "abs_max": 0.9025078735164493,
            "n_non_finite": 0,
            "sample": [
                [0.0, 0.786932145996389],
                [0.5564879497286803, 0.5564369184553719],
                [0.7869812951406054, 4.818428722308927e-17],
                [0.5564879497286802, -0.5564369184553721],
                [9.638023174804209e-17, -0.786932145996389],
                [-0.5564879497286802, -0.5564369184553721],
                [-0.7869812951406054, 4.818428722308927e-17],
                [-0.5564879497286803, 0.5564369184553719],
                [-6.592519057047366e-37, 0.1491521048809516],
                [0.14916687747075894, 9.13292525159712e-18],
                [1.8267678339669237e-17, -0.1491521048809516],
                [-0.14916687747075894, 9.13292525159712e-18],
                [0.4951122158556473, 0.4950889571720242],
                [0.49511221585564746, -0.4950889571720242],
                [-0.49511221585564746, -0.4950889571720242],
                [-0.4951122158556473, 0.4950889571720242],
            ],
        },
    },
    "hst": {
        "Gaussian": {
            "abs_sum": 16710.89555761894,
            "abs_max": 1.0045988177091016,
            "n_non_finite": 0,
            "sample": [
                [-0.08522848494462226, 0.8223777453345469],
                [0.5347249918530897, 0.5347249214028542],
                [0.8223777453345469, -0.08522848494462215],
                [0.6240102336059056, -0.6240102336059056],
                [0.08522848494462248, -0.8223777453345469],
                [-0.5347249918530889, -0.534724921402854],
                [-0.8223777453345469, 0.08522848494462237],
                [-0.6240102336059056, 0.6240102336059056],
                [-0.0165657619170127, 0.14931861195098112],
                [0.14931861195098112, -0.016565761917012686],
                [0.016565761917012825, -0.149318611950981],
                [-0.149318611950981, 0.016565761917012797],
                [0.6179239070956012, 0.6179239070956013],
                [0.5837883046591865, -0.5837883046591864],
                [-0.6179239070956013, -0.6179239070956012],
                [-0.5837883046591865, 0.5837883046591864],
            ],
        },
        "Gaussian_sph_case": {
            "abs_sum": 14360.33826028308,
            "abs_max": 0.9025078735164493,
            "n_non_finite": 0,
            "sample": [
                [0.0, 0.786932145996389],
                [0.5564879497286803, 0.5564369184553719],
                [0.7869812951406054, 4.818428722308927e-17],
                [0.5564879497286802, -0.5564369184553721],
                [9.638023174804209e-17, -0.786932145996389],
                [-0.5564879497286802, -0.5564369184553721],
                [-0.7869812951406054, 4.818428722308927e-17],
                [-0.5564879497286803, 0.5564369184553719],
                [-6.592519057047366e-37, 0.1491521048809516],
                [0.14916687747075894, 9.13292525159712e-18],
                [1.8267678339669237e-17, -0.1491521048809516],
                [-0.14916687747075894, 9.13292525159712e-18],
                [0.4951122158556473, 0.4950889571720242],
                [0.49511221585564746, -0.4950889571720242],
                [-0.49511221585564746, -0.4950889571720242],
                [-0.4951122158556473, 0.4950889571720242],
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
