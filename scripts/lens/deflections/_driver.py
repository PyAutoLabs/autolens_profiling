"""Shared driver for the deflection-angle profiling cells.

Every cell under ``scripts/lens/deflections/`` is a thin file: a prologue, an
``EXPECTED`` pin block, and one call to :func:`run`. Everything measurable lives
here — grid construction, the timing loop, the separate cProfile pass, the pin
check, and the JSON + PNG write — so the three cells cannot drift apart in
methodology and a change to the method is a one-file change.

What is measured
----------------

For each profile, three **library** callables are timed separately (never a
transcription of their internals — the point is the cost users actually pay):

a. ``profile.deflections_yx_2d_from(grid=dataset.grids.pixelization)``
   — a ``Grid2D`` at sub-size 1, i.e. one deflection per masked image pixel.
   This is the ray-trace grid the mapper consumes.
b. ``profile.deflections_yx_2d_from(grid=dataset.grids.lp.over_sampled)``
   — the ``Grid2DIrregular`` companion of the light-profile grid, carrying the
   radial-bin over-sampling ([4, 2, 1] inside 0.3" / 0.6"). Slightly larger than
   (a); the ratio between the two is the over-sampling tax.
c. ``al.Tracer([Galaxy(z=0.5, mass=profile), Galaxy(z=1.0)]).traced_grid_2d_list_from(...)``
   — the same deflection through the two-plane tracer, so the wrapper cost
   (grid bookkeeping, plane assembly, the subtraction) is visible next to the
   raw call rather than assumed.

The grid construction mirrors ``scripts/imaging/likelihood_breakdown/
pixelization_numba.py`` exactly (circular 3.5" mask, radial-bin over-sampling,
``over_sample_size_pixelization=1``), so a deflection number here is directly
comparable to the ray-trace row of that cell's breakdown.

cProfile
--------

A **separate** pass per profile cProfiles call (a) — the profiler's tracing
overhead inflates every number it reports, so it is never allowed near the
headline timings. The stored ``cprofile_top`` is the top 12 functions by
cumulative time, per call, purely as *attribution*: which library function the
time is charged to, not how long it takes.

Pins
----

Timings drift with the machine; the numbers a profile computes must not. Each
run pins three things per profile — ``abs_sum`` and ``abs_max`` over the full
grid-(a) deflection field, plus a 16-coordinate ``(y, x)`` ``sample`` on a
dedicated ``Grid2DIrregular`` at fixed arcsec positions — and checks them at
rtol 1e-6. Drift is **recorded, never adjudicated** (the boundary rule in
``results/notes/design_lock_in.md``): a changed number flags the run's timings
as non-comparable, it does not fail the job.

On the first run for an instrument the cell has no pins, so the driver fills its
``EXPECTED`` block in place and says so. ``--repin`` re-fills it deliberately:
it requires ``--repin-reason``, prints the old/new diff, refuses any value moving
by more than ``--repin-max-shift`` relative unless ``--repin-force``, and records
``pin_provenance`` in the result JSON.
"""

from __future__ import annotations

import argparse
import cProfile
import datetime as _datetime
import json
import math
import os
import pstats
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _profiles import PROFILES  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    check_pinned_vector,
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)
from instruments.imaging import INSTRUMENTS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INSTRUMENT = "hst"
INSTRUMENT_CHOICES = ("hst", "euclid")

MASK_RADIUS = 3.5
SUB_SIZE_LIST = [4, 2, 1]
RADIAL_LIST = [0.3, 0.6]

PIN_RTOL = 1e-6

LENS_REDSHIFT = 0.5
SOURCE_REDSHIFT = 1.0

# The 16 fixed pin coordinates, in arcsec, as (y, x). Eight sit on r = 1.0"
# at 45-degree steps (the einstein-radius annulus, where every profile is
# well-conditioned), four on r = 0.15" (inside the over-sampled core, where the
# cusp/core behaviour of the NFW family differs), and four on r = 2.8" (the
# outer mask, where the wings decide the ray-trace). Written out literally
# rather than generated so the pin is legible and cannot silently change with a
# refactor of the generating expression.
_R1 = 0.7071067811865476  # 1.0 / sqrt(2)
_R28 = 1.9798989873223332  # 2.8 / sqrt(2)

PIN_COORDINATES: list[tuple[float, float]] = [
    # r = 1.0", 45-degree steps
    (0.0, 1.0),
    (_R1, _R1),
    (1.0, 0.0),
    (_R1, -_R1),
    (0.0, -1.0),
    (-_R1, -_R1),
    (-1.0, 0.0),
    (-_R1, _R1),
    # r = 0.15", inside the over-sampled core
    (0.0, 0.15),
    (0.15, 0.0),
    (0.0, -0.15),
    (-0.15, 0.0),
    # r = 2.8", the outer mask
    (_R28, _R28),
    (_R28, -_R28),
    (-_R28, -_R28),
    (-_R28, _R28),
]

# Sentinels delimiting the pin block a cell keeps in its own source. ``--repin``
# (and the first-run auto-fill) rewrite everything between them.
PIN_BEGIN = "# >>> BEGIN EXPECTED (generated — rewrite with --repin) >>>"
PIN_END = "# <<< END EXPECTED <<<"

CPROFILE_TOP_N = 12
CPROFILE_NOTE = (
    "Attribution only, from a SEPARATE cProfile pass over cprofile_repeats calls "
    "of deflections_yx_2d_from(grid=grids.pixelization). Every number here is "
    "inflated by the profiler's per-call tracing overhead and by the profiler's "
    "own bookkeeping; they say WHICH function the time is charged to, not how "
    "long it takes. The headline timings are grid2d_s / irregular_s / tracer_s, "
    "measured with the profiler off."
)


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class DeflectionCLI:
    """The shared sweep flags plus the deflection cells' own flags."""

    def __init__(self, base, args):
        self.base = base
        self.instrument = (base.instrument or DEFAULT_INSTRUMENT).lower()
        self.n_repeats = int(args.n_repeats)
        self.cprofile_repeats = int(args.cprofile_repeats)
        self.repin = bool(args.repin)
        self.repin_reason = args.repin_reason
        self.repin_max_shift = float(args.repin_max_shift)
        self.repin_force = bool(args.repin_force)


def parse_cli() -> DeflectionCLI:
    """``parse_profile_cli`` for the shared flags + the deflection-cell flags.

    ``parse_known_args`` throughout, matching every other cell in the repo: the
    sweep driver passes flags this script does not know about and vice versa.
    """
    base = parse_profile_cli()

    parser = argparse.ArgumentParser(
        description="Deflection-angle profiling cell flags.", allow_abbrev=False
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Timed repeats per (profile, grid) after one warm-up call; the median is recorded.",
    )
    parser.add_argument(
        "--cprofile-repeats",
        type=int,
        default=5,
        help="Calls in the separate cProfile attribution pass (never the headline timing).",
    )
    parser.add_argument(
        "--repin",
        action="store_true",
        help="Re-measure and rewrite this cell's EXPECTED pin block in place.",
    )
    parser.add_argument(
        "--repin-reason",
        default=None,
        help="Why the pins are moving (required with --repin; stored as pin_provenance).",
    )
    parser.add_argument(
        "--repin-max-shift",
        type=float,
        default=1e-3,
        help="Refuse the re-pin if any pinned value moves by more than this, relative.",
    )
    parser.add_argument(
        "--repin-force",
        action="store_true",
        help="Allow a re-pin that exceeds --repin-max-shift.",
    )
    args, _unknown = parser.parse_known_args()
    return DeflectionCLI(base, args)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def build_dataset(instrument: str, workspace_root: Path):
    """Load + mask + over-sample the imaging dataset for ``instrument``.

    Identical to the grid construction in ``scripts/imaging/likelihood_breakdown/
    pixelization_numba.py``, so the deflection cost measured here is the cost
    that cell's ray-trace step pays.
    """
    if instrument not in INSTRUMENT_CHOICES:
        raise SystemExit(
            f"--instrument {instrument!r} not supported by the deflection cells; "
            f"choose one of {', '.join(INSTRUMENT_CHOICES)}."
        )

    pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
    dataset_path = Path("dataset") / "imaging" / instrument

    auto_simulate_if_missing(
        dataset_path,
        dataset_type="imaging",
        instrument=instrument,
        workspace_root=workspace_root,
    )

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=MASK_RADIUS,
    )
    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=SUB_SIZE_LIST,
        radial_list=RADIAL_LIST,
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )
    return dataset, pixel_scale


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _n_points(grid) -> int:
    """Number of (y, x) coordinates in a ``Grid2D`` / ``Grid2DIrregular``."""
    return int(np.asarray(getattr(grid, "array", grid)).shape[0])


def _median_seconds(fn, n_repeats: int) -> float:
    """One warm-up call, then the median of ``n_repeats`` timed calls."""
    fn()
    samples = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def _cprofile_top(fn, n_repeats: int, top: int = CPROFILE_TOP_N) -> list[dict]:
    """Top-``top`` functions by cumulative time, per call, from a separate pass."""
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_repeats):
        fn()
    profiler.disable()

    rows = []
    for (filename, lineno, funcname), entry in pstats.Stats(profiler).stats.items():
        _primitive_calls, ncalls, tottime, cumtime, _callers = entry
        rows.append(
            {
                "func": f"{Path(filename).name}:{lineno}({funcname})",
                "ncalls": ncalls / n_repeats,
                "tottime_s": tottime / n_repeats,
                "cumtime_s": cumtime / n_repeats,
            }
        )
    rows.sort(key=lambda r: -r["cumtime_s"])
    return rows[:top]


def _pins_from(alpha_grid2d, alpha_sample) -> dict:
    """The pin block for one profile: three scalars plus the 16-coordinate sample.

    ``abs_sum`` / ``abs_max`` are **nan-aware** reductions, and the non-finite
    entries they skip are pinned separately as ``n_non_finite``. At least one
    profile in this set needs that: ``PowerLawSph`` returns a deflection field
    with two non-finite entries on both the hst and euclid grids, and a plain
    ``sum`` would collapse its whole pin to NaN — hiding every other value behind
    one bad point instead of recording it. Pinning the count keeps both facts:
    how many points went non-finite, and what the rest of the field sums to.
    """
    arr = np.asarray(alpha_grid2d, dtype=float)
    finite = np.isfinite(arr)
    abs_arr = np.abs(arr)
    sample = np.asarray(alpha_sample, dtype=float).reshape(-1, 2)
    return {
        "abs_sum": float(np.sum(abs_arr[finite])) if finite.any() else 0.0,
        "abs_max": float(np.max(abs_arr[finite])) if finite.any() else 0.0,
        "n_non_finite": int(arr.size - int(finite.sum())),
        "sample": [[float(y), float(x)] for y, x in sample],
    }


def measure_profile(spec, grids, sample_grid, cli) -> tuple[dict, dict]:
    """Time + cProfile + pin one profile. Returns ``(record, pins)``."""
    profile = spec.build()
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=LENS_REDSHIFT, mass=profile),
            al.Galaxy(redshift=SOURCE_REDSHIFT),
        ]
    )

    grid2d, irregular = grids

    def _call_grid2d():
        return profile.deflections_yx_2d_from(grid=grid2d)

    def _call_irregular():
        return profile.deflections_yx_2d_from(grid=irregular)

    def _call_tracer():
        return tracer.traced_grid_2d_list_from(grid=grid2d)

    grid2d_s = _median_seconds(_call_grid2d, cli.n_repeats)
    irregular_s = _median_seconds(_call_irregular, cli.n_repeats)
    tracer_s = _median_seconds(_call_tracer, cli.n_repeats)

    cprofile_top = _cprofile_top(_call_grid2d, cli.cprofile_repeats)

    pins = _pins_from(
        _call_grid2d(),
        profile.deflections_yx_2d_from(grid=sample_grid),
    )

    record = {
        "family": spec.family,
        "params": spec.json_params,
        "grid2d_s": grid2d_s,
        "irregular_s": irregular_s,
        "tracer_s": tracer_s,
        "grid2d_n_points": _n_points(grid2d),
        "irregular_n_points": _n_points(irregular),
        "tracer_n_points": _n_points(grid2d),
        "tracer_over_raw": tracer_s / grid2d_s if grid2d_s else None,
        "cprofile_repeats": cli.cprofile_repeats,
        "cprofile_note": CPROFILE_NOTE,
        "cprofile_top": cprofile_top,
    }
    return record, pins


# ---------------------------------------------------------------------------
# Pin block rendering / rewriting
# ---------------------------------------------------------------------------


def _py_float(value) -> str:
    """A float as a Python source literal that round-trips, NaN and inf included.

    ``repr(float("nan"))`` is ``"nan"``, which is not a name any module defines —
    writing it straight into a pin block produces a cell that raises
    ``NameError`` on its next run.
    """
    v = float(value)
    if math.isnan(v):
        return 'float("nan")'
    if math.isinf(v):
        return 'float("inf")' if v > 0 else 'float("-inf")'
    return repr(v)


def _render_expected(expected: dict) -> str:
    """Render the EXPECTED literal in ruff-format canonical shape.

    Magic trailing commas everywhere keep the block exploded exactly as written,
    so ``ruff format --check`` is stable across a re-pin.
    """
    if not expected:
        return "EXPECTED: dict[str, dict[str, dict]] = {}"

    lines = ["EXPECTED: dict[str, dict[str, dict]] = {"]
    for instrument in sorted(expected):
        lines.append(f'    "{instrument}": {{')
        for name in expected[instrument]:
            pins = expected[instrument][name]
            lines.append(f'        "{name}": {{')
            lines.append(f'            "abs_sum": {_py_float(pins["abs_sum"])},')
            lines.append(f'            "abs_max": {_py_float(pins["abs_max"])},')
            lines.append(f'            "n_non_finite": {int(pins["n_non_finite"])},')
            lines.append('            "sample": [')
            for y, x in pins["sample"]:
                lines.append(f"                [{_py_float(y)}, {_py_float(x)}],")
            lines.append("            ],")
            lines.append("        },")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def _write_expected(cell_path: Path, expected: dict) -> None:
    """Replace the cell's pin block in place, then best-effort ``ruff format``."""
    text = cell_path.read_text()
    if PIN_BEGIN not in text or PIN_END not in text:
        raise RuntimeError(
            f"{cell_path.name} has no EXPECTED pin block sentinels "
            f"({PIN_BEGIN!r} / {PIN_END!r}); cannot rewrite it."
        )
    head, rest = text.split(PIN_BEGIN, 1)
    _old, tail = rest.split(PIN_END, 1)
    cell_path.write_text(f"{head}{PIN_BEGIN}\n{_render_expected(expected)}\n{PIN_END}{tail}")
    try:
        subprocess.run(
            ["ruff", "format", str(cell_path)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        print(
            "  NOTE: `ruff format` unavailable — re-run it over "
            f"{cell_path.name} before committing the new pins."
        )


def _pin_values(pins: dict) -> list[float]:
    return [
        float(pins["abs_sum"]),
        float(pins["abs_max"]),
        float(pins["n_non_finite"]),
    ] + [float(v) for pair in pins["sample"] for v in pair]


def _relative(new_v: np.ndarray, old_v: np.ndarray) -> np.ndarray:
    """Element-wise relative move, NaN-aware (NaN vs NaN matches, NaN vs number is inf)."""
    with np.errstate(invalid="ignore"):
        rel = np.abs(new_v - old_v) / np.maximum(np.abs(old_v), 1e-300)
    rel = np.where(np.isnan(new_v) & np.isnan(old_v), 0.0, rel)
    return np.where(np.isnan(rel), np.inf, rel)


def _max_relative_shift(old: dict, new: dict) -> tuple[float, str]:
    """Largest relative move between two pin blocks, and the label that moved."""
    worst, worst_label = 0.0, ""
    for name, new_pins in new.items():
        old_pins = old.get(name)
        if old_pins is None:
            continue  # a newly pinned profile has nothing to move away from
        old_v = np.asarray(_pin_values(old_pins), dtype=float)
        new_v = np.asarray(_pin_values(new_pins), dtype=float)
        if old_v.shape != new_v.shape:
            return float("inf"), f"{name} (pin shape changed)"
        rel = _relative(new_v, old_v)
        if float(rel.max()) > worst:
            worst, worst_label = float(rel.max()), name
    return worst, worst_label


def _print_pin_diff(old: dict, new: dict) -> None:
    print("\n--- Pin diff (old -> new) ---")
    for name in new:
        old_pins = old.get(name)
        if old_pins is None:
            print(f"  {name:<20s} NEW      abs_sum {new[name]['abs_sum']!r}")
            continue
        for key in ("abs_sum", "abs_max", "n_non_finite"):
            o, n = float(old_pins[key]), float(new[name][key])
            rel = float(_relative(np.array([n]), np.array([o]))[0])
            print(f"  {name:<20s} {key:<13s} {o!r} -> {n!r}  (rel {rel:.3e})")
        o_s = np.asarray(old_pins["sample"], dtype=float).ravel()
        n_s = np.asarray(new[name]["sample"], dtype=float).ravel()
        if o_s.shape == n_s.shape:
            rel = float(np.max(_relative(n_s, o_s)))
            print(f"  {name:<20s} {'sample':<13s} max rel {rel:.3e} over {o_s.size} values")
        else:
            print(f"  {name:<20s} {'sample':<13s} SHAPE CHANGE {o_s.size} -> {n_s.size}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _write_png(chart_path: Path, records: dict, *, cell: str, instrument: str, al_version: str):
    names = list(records)
    series = [
        ("Grid2D (grids.pixelization)", "grid2d_s", "#4C72B0"),
        ("Grid2DIrregular (lp.over_sampled)", "irregular_s", "#DD8452"),
        ("Tracer.traced_grid_2d_list_from", "tracer_s", "#55A868"),
    ]

    fig, ax = plt.subplots(figsize=(10, 1.1 * len(names) + 2.6))
    height = 0.26
    for offset, (label, key, colour) in zip((-height, 0.0, height), series):
        values = [records[n][key] for n in names]
        positions = [i + offset for i in range(len(names))]
        bars = ax.barh(
            positions, values, height=height, color=colour, edgecolor="white", label=label
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() * 1.06,
                bar.get_y() + bar.get_height() / 2,
                f"{value * 1e3:.2f} ms",
                va="center",
                fontsize=7.5,
            )

    ax.set_xscale("log")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time per call (s, log scale)", fontsize=11)
    # Below the axes, not inside them: with a log axis the longest bar reaches
    # the right edge on every cell, and an in-axes legend sits on its label.
    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        frameon=False,
    )
    ax.margins(x=0.30)
    fig.suptitle(
        f"Numpy deflection angles — {cell} — {instrument.upper()}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"AutoLens v{al_version}  |  "
        f"{records[names[0]]['grid2d_n_points']} grid points  |  "
        f"{records[names[0]]['irregular_n_points']} over-sampled  |  "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(*, cell: str, profiles: list[str], cell_path: Path, expected: dict) -> None:
    """Measure ``profiles`` for one cell and write the JSON + PNG artifact pair."""
    cli = parse_cli()
    instrument = cli.instrument
    workspace_root = _profiling_root()

    if cli.repin and not cli.repin_reason:
        raise SystemExit("--repin requires --repin-reason (recorded as pin_provenance).")

    unknown = [name for name in profiles if name not in PROFILES]
    if unknown:
        raise SystemExit(f"Unknown profile(s) in cell {cell!r}: {', '.join(unknown)}")

    print(f"\n--- Dataset loading & masking [{instrument}] ---")
    dataset, pixel_scale = build_dataset(instrument, workspace_root)

    grid2d = dataset.grids.pixelization
    irregular = dataset.grids.lp.over_sampled
    sample_grid = al.Grid2DIrregular(values=PIN_COORDINATES)

    n_grid2d = _n_points(grid2d)
    n_irregular = _n_points(irregular)

    print("\n--- Configuration (determines run time) ---")
    print(f"  Cell:                    {cell}")
    print(f"  Instrument:              {instrument}")
    print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
    print(f"  Mask radius:             {MASK_RADIUS} arcsec")
    print(f"  Grid2D points:           {n_grid2d}")
    print(f"  Irregular points:        {n_irregular}")
    print(f"  Timed repeats:           {cli.n_repeats} (median; 1 warm-up)")
    print(f"  cProfile repeats:        {cli.cprofile_repeats} (separate pass)")
    print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")

    records: dict[str, dict] = {}
    measured_pins: dict[str, dict] = {}

    print("\n--- Timings (seconds per call, median) ---")
    for name in profiles:
        spec = PROFILES[name]
        record, pins = measure_profile(spec, (grid2d, irregular), sample_grid, cli)
        records[name] = record
        measured_pins[name] = pins
        print(
            f"  {name:<20s} grid2d {record['grid2d_s']:10.6f} s   "
            f"irregular {record['irregular_s']:10.6f} s   "
            f"tracer {record['tracer_s']:10.6f} s   "
            f"(tracer/raw {record['tracer_over_raw']:.2f}x)"
        )

    # --- Pin resolution -----------------------------------------------------

    today = _datetime.date.today().isoformat()
    stored = expected.get(instrument, {})
    pin_provenance = None
    pinned_drift: list = []
    exit_code = 0

    if cli.repin:
        shift, label = _max_relative_shift(stored, measured_pins)
        _print_pin_diff(stored, measured_pins)
        if shift > cli.repin_max_shift and not cli.repin_force:
            print(
                f"\n  RE-PIN REFUSED — {label or 'the largest pinned move'} is {shift:.3e} "
                f"relative, above "
                f"--repin-max-shift {cli.repin_max_shift:g}. A move this large is a "
                f"changed computation, not a re-measurement: establish why first, then "
                f"re-run with --repin-force if the new values are the intended ones. "
                f"The existing pins are untouched."
            )
            exit_code = 1
        else:
            expected[instrument] = measured_pins
            _write_expected(cell_path, expected)
            pin_provenance = {
                "reason": cli.repin_reason,
                "date": today,
                "max_shift": shift,
            }
            print(
                f"\n  Pins REWRITTEN in {cell_path.name} for {instrument} "
                f"(max shift {shift:.3e}): {cli.repin_reason}"
            )
            stored = measured_pins
    elif not stored:
        expected[instrument] = measured_pins
        _write_expected(cell_path, expected)
        pin_provenance = {
            "reason": f"first run — no pins existed for instrument {instrument}",
            "date": today,
            "max_shift": None,
        }
        print(
            f"\n  FIRST RUN for {instrument}: no pins existed, so the EXPECTED block in "
            f"{cell_path.name} has been filled from this run. Every subsequent run checks "
            f"against it; nothing was verified on this one."
        )
        stored = measured_pins

    if pin_provenance is None:
        print("\n--- Pinned-value checks (rtol 1e-6) ---")
        for name in profiles:
            pins, got = stored.get(name), measured_pins[name]
            if pins is None:
                print(f"  {name:<20s} SKIPPED (no pinned values)")
                continue
            for key in ("abs_sum", "abs_max", "n_non_finite"):
                record = check_pinned(
                    got[key], float(pins[key]), label=f"{name}.{key}", rtol=PIN_RTOL
                )
                if record is not None:
                    pinned_drift.append(record)
            record = check_pinned_vector(
                got["sample"], pins["sample"], label=f"{name}.sample", rtol=PIN_RTOL
            )
            if record is not None:
                pinned_drift.append(record)
        if not pinned_drift:
            print("  Pinned-value checks PASSED (recorded in the result JSON).")

    # --- Write --------------------------------------------------------------

    al_version = al.__version__

    summary = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "instrument": instrument,
        "cell": cell,
        "configuration": {
            "pixel_scale_arcsec": pixel_scale,
            "mask_radius_arcsec": MASK_RADIUS,
            "grid2d_n_points": n_grid2d,
            "irregular_n_points": n_irregular,
            "sub_size_list": list(SUB_SIZE_LIST),
            "radial_list": list(RADIAL_LIST),
            "n_repeats": cli.n_repeats,
            "cprofile_repeats": cli.cprofile_repeats,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
            "use_jax": False,
            "xp": "numpy",
        },
        "profiles": records,
        "pinned_expected": stored or None,
        "pinned_drift": pinned_drift,
        "pin_provenance": pin_provenance,
    }

    dict_path, chart_path = resolve_output_paths(
        cli.base,
        default_dir=workspace_root / "results" / "lens" / "deflections",
        default_basename=f"{cell}_summary_{instrument}_v{al_version}",
    )
    dict_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results dict saved to: {dict_path}")

    _write_png(chart_path, records, cell=cell, instrument=instrument, al_version=al_version)
    print(f"  Bar chart saved to:    {chart_path}")

    if exit_code:
        sys.exit(exit_code)
