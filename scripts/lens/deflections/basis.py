"""
Numpy CPU Profiling: Deflection Angles — MGE Basis (fixed-geometry memo)
=======================================================================

Times ``deflections_yx_2d_from`` for a ``Basis`` of 30 light-and-mass Gaussians with
log-spaced widths — the shape an MGE lens light takes in the SLaM ``mass_light_dark``
stage, where every Gaussian's centre, ellipticity, intensity and sigma are fixed by the
light stage and the whole stack shares **one** ``mass_to_light_ratio``.

This is the cell the ``gaussian-deflections-precompute`` epic (PyAutoGalaxy#601) reports
through. Phase 1 added a numpy-path cross-evaluation memo
(``autogalaxy/profiles/mass/abstract/deflections_memo.py``): each Gaussian's unit-ratio
Faddeeva field is computed once per (parameters, grid) and later evaluations return
``mass_to_light_ratio x field``, so a 30-Gaussian basis collapses to 30 multiply-adds.

The headline ``grid2d_s`` / ``irregular_s`` / ``tracer_s`` timings below are, as in every
other cell here, measured with the memo **suspended** (``_driver.measure_profile`` holds
``memo_disabled()``): the driver's job is the uncached per-call cost of computing a
deflection, which must stay comparable across the library's history. The memo's effect
is measured deliberately in the block after the driver call, which reports memo-off
against memo-on for the same basis and adds two things the timings alone cannot show:

1. **A ``_wofz`` call-count witness** — the Faddeeva function is counted across three
   consecutive evaluations. With fixed geometry it must be called on the first
   evaluation and **exactly zero** times on the second and third. Two controls prove
   the counter is live: varying a geometry parameter per call, and the kill switch.
2. **The memo's memory footprint** against its byte cap.

Method, pin policy and the ``--repin`` contract all live in ``_driver.py``; the fiducial
parameters live in ``_profiles.py``. This file is deliberately thin: a prologue, the
pinned expectations, one driver call, and the memo witness.

Output
------
``results/lens/deflections/basis_summary_<instrument>_v<version>.{json,png}``
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

import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from _profiles import PROFILES  # noqa: E402
from autogalaxy.profiles.mass.abstract import deflections_memo as _memo  # noqa: E402
from autogalaxy.profiles.mass.abstract import mge as _mge  # noqa: E402

# The pinned deflection values for this cell, keyed by instrument then profile:
# ``abs_sum`` / ``abs_max`` over the full Grid2D deflection field, and a
# 16-coordinate (y, x) ``sample`` at the fixed arcsec positions in
# ``_driver.PIN_COORDINATES``. Checked at rtol 1e-6 on every run. Filled
# automatically on the first run for an instrument; moved only via
# ``--repin --repin-reason "..."``. Do not hand-edit.
# >>> BEGIN EXPECTED (generated — rewrite with --repin) >>>
EXPECTED: dict[str, dict[str, dict]] = {
    "euclid": {
        "Basis_mge_30": {
            "abs_sum": 64945.02779197226,
            "abs_max": 14.632232306435608,
            "n_non_finite": 0,
            "sample": [
                [-0.9777613190509087, 10.02843272927424],
                [6.576187656444237, 6.576187656444234],
                [10.028432729274243, -0.9777613190509074],
                [7.578483087653899, -7.5784830876538996],
                [0.9777613190509087, -10.02843272927424],
                [-6.576187656444237, -6.576187656444234],
                [-10.028432729274243, 0.9777613190509074],
                [-7.578483087653899, 7.5784830876538996],
                [-0.30247300400012683, 2.9049133883830156],
                [2.9049133883830156, -0.3024730040001268],
                [0.30247300400012683, -2.9049133883830156],
                [-2.9049133883830156, 0.3024730040001268],
                [9.76114474587484, 9.761144745874843],
                [10.574485349676419, -10.57448534967642],
                [-9.76114474587484, -9.761144745874843],
                [-10.574485349676419, 10.57448534967642],
            ],
        },
    },
    "hst": {
        "Basis_mge_30": {
            "abs_sum": 259587.23590204224,
            "abs_max": 14.63398899797888,
            "n_non_finite": 0,
            "sample": [
                [-0.9777613190509087, 10.02843272927424],
                [6.576187656444237, 6.576187656444234],
                [10.028432729274243, -0.9777613190509074],
                [7.578483087653899, -7.5784830876538996],
                [0.9777613190509087, -10.02843272927424],
                [-6.576187656444237, -6.576187656444234],
                [-10.028432729274243, 0.9777613190509074],
                [-7.578483087653899, 7.5784830876538996],
                [-0.30247300400012683, 2.9049133883830156],
                [2.9049133883830156, -0.3024730040001268],
                [0.30247300400012683, -2.9049133883830156],
                [-2.9049133883830156, 0.3024730040001268],
                [9.76114474587484, 9.761144745874843],
                [10.574485349676419, -10.57448534967642],
                [-9.76114474587484, -9.761144745874843],
                [-10.574485349676419, 10.57448534967642],
            ],
        },
    },
}
# <<< END EXPECTED <<<

_driver.run(
    cell="basis",
    profiles=["Basis_mge_30"],
    cell_path=_Path(__file__).resolve(),
    expected=EXPECTED,
)


# ===========================================================================
# Memo witness — memo-on vs memo-off, the _wofz call counter, and the byte cap
# ===========================================================================

WITNESS_REPEATS = 20


def _median_seconds(fn, n_repeats: int) -> float:
    """One warm-up call, then the median of ``n_repeats`` timed calls."""
    fn()
    samples = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def _memo_env(enabled: bool) -> None:
    os.environ["AUTOGALAXY_DEFLECTIONS_MEMO"] = "1" if enabled else "0"
    _memo.memo_clear()


class _WofzCounter:
    """Counts ``mge._wofz`` calls by wrapping it for the duration of the block.

    Wrapped here in the cell, never in the library: the witness must not be able to
    change what it measures.
    """

    def __enter__(self):
        self.count = 0
        self._original = _mge._wofz

        def counted(z, xp=np):
            self.count += 1
            return self._original(z, xp=xp)

        _mge._wofz = counted
        return self

    def __exit__(self, *exception):
        _mge._wofz = self._original
        return False


def _wofz_calls_per_evaluation(build, grid, n_evaluations: int = 3) -> list:
    """``_wofz`` calls made by each of ``n_evaluations`` consecutive deflection calls."""
    counts = []
    with _WofzCounter() as counter:
        for evaluation in range(n_evaluations):
            counter.count = 0
            build(evaluation).deflections_yx_2d_from(grid=grid)
            counts.append(counter.count)
    return counts


_cli = _driver.parse_cli()
_instrument = _cli.instrument
_dataset, _ = _driver.build_dataset(_instrument, _profiling_root())
_grid = _dataset.grids.pixelization

_spec = PROFILES["Basis_mge_30"]
_ratio = _spec.params["mass_to_light_ratio"]


def _basis(mass_to_light_ratio=_ratio, axis_ratio=_spec.params["axis_ratio"]):
    params = dict(_spec.params)
    params["mass_to_light_ratio"] = mass_to_light_ratio
    params["axis_ratio"] = axis_ratio
    return _spec.factory(**params)


print("\n" + "=" * 74)
print(f"FIXED-GEOMETRY DEFLECTION MEMO — WITNESS — {_instrument.upper()}")
print("=" * 74)

# --- 1. memo-on vs memo-off, same basis, same grid --------------------------

# A throwaway pass first: the first basis evaluation in a process pays scipy/numba
# import and page-fault costs that would otherwise be charged to whichever leg runs
# first, and this machine's medians move by ~50% without it.
_memo_env(False)
_basis().deflections_yx_2d_from(grid=_grid)

_memo_env(False)
_off_s = _median_seconds(lambda: _basis().deflections_yx_2d_from(grid=_grid), WITNESS_REPEATS)

_memo_env(True)
_on_s = _median_seconds(lambda: _basis().deflections_yx_2d_from(grid=_grid), WITNESS_REPEATS)

# The sampler's case: the ratio moves every evaluation, the geometry does not.
_memo_env(True)
_basis(mass_to_light_ratio=1.0).deflections_yx_2d_from(grid=_grid)
_ratio_cycle = iter(np.linspace(0.5, 2.5, WITNESS_REPEATS + 1))
_free_ratio_s = _median_seconds(
    lambda: _basis(mass_to_light_ratio=float(next(_ratio_cycle))).deflections_yx_2d_from(
        grid=_grid
    ),
    WITNESS_REPEATS,
)

print(
    "\n--- Basis deflections, Grid2D, median of "
    f"{WITNESS_REPEATS} calls ({_grid.array.shape[0]} points) ---"
)
print(f"  memo OFF                         {_off_s * 1e3:10.3f} ms")
print(f"  memo ON  (geometry + ratio fixed){_on_s * 1e3:10.3f} ms   ({_off_s / _on_s:.1f}x)")
print(
    f"  memo ON  (ratio free per call)   {_free_ratio_s * 1e3:10.3f} ms   "
    f"({_off_s / _free_ratio_s:.1f}x)"
)

# --- 2. the _wofz call-count witness ----------------------------------------

_memo_env(True)
_fixed_counts = _wofz_calls_per_evaluation(lambda i: _basis(), _grid)

_memo_env(True)
_varying_counts = _wofz_calls_per_evaluation(lambda i: _basis(axis_ratio=0.8 - 0.01 * i), _grid)

_memo_env(False)
_killed_counts = _wofz_calls_per_evaluation(lambda i: _basis(), _grid)

_memo_env(True)

print("\n--- _wofz call counts over 3 consecutive evaluations ---")
print(f"  fixed geometry, memo on          {_fixed_counts}")
print(f"  geometry varied per call         {_varying_counts}   (control)")
print(f"  fixed geometry, memo off         {_killed_counts}   (kill-switch control)")

assert _fixed_counts[1] == 0 and _fixed_counts[2] == 0, (
    f"fixed-geometry evaluations 2 and 3 still called _wofz: {_fixed_counts}"
)
assert all(count > 0 for count in _varying_counts), (
    f"varying-geometry control called _wofz zero times: {_varying_counts}"
)
assert all(count > 0 for count in _killed_counts), (
    f"kill-switch control called _wofz zero times: {_killed_counts}"
)

# --- 3. the memo is bounded --------------------------------------------------

_memo_env(True)
for _repeat in range(3):
    _basis(mass_to_light_ratio=1.0 + 0.1 * _repeat).deflections_yx_2d_from(grid=_grid)

_stats = _memo.memo_stats()
_cap_bytes = _memo.memo_max_bytes()

print("\n--- Memo footprint ---")
print(f"  entries                          {_stats['entries']}")
print(f"  bytes                            {_stats['bytes'] / 1024**2:.2f} MB")
print(f"  cap                              {_cap_bytes / 1024**2:.0f} MB")
print(
    f"  hits / misses / evictions        {_stats['hits']} / {_stats['misses']} / "
    f"{_stats['evictions']}"
)

assert _stats["bytes"] <= _cap_bytes, (
    f"memo holds {_stats['bytes']} bytes, above its {_cap_bytes}-byte cap"
)

# --- 4. the memo does not change the answer ----------------------------------

_memo_env(False)
_answer_off = np.asarray(_basis(mass_to_light_ratio=1.7).deflections_yx_2d_from(grid=_grid).array)

_memo_env(True)
_basis(mass_to_light_ratio=1.0).deflections_yx_2d_from(grid=_grid)
_answer_on = np.asarray(_basis(mass_to_light_ratio=1.7).deflections_yx_2d_from(grid=_grid).array)

_max_rel = float(np.max(np.abs(_answer_on - _answer_off) / np.maximum(np.abs(_answer_off), 1e-300)))
print(f"\n  memo-on vs memo-off, max relative difference: {_max_rel:.3e}")
assert _max_rel < 1e-12, f"memo changed the deflection field by {_max_rel:.3e} relative"

print("=" * 74)
