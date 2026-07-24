"""
Quick-Update Profiling: Imaging MGE (HST)
=========================================

Profiles the quick-update visualization path for an imaging model-fit
with an **MGE lens + MGE source** — the default ``start_here.py`` model.

This is representative of what users actually see during modeling: 20
MGE Gaussians per galaxy (40 total), ``Isothermal`` mass + shear.

Usage::

    cd autolens_profiling
    python quick_update/imaging.py
    python quick_update/imaging.py --instrument euclid
    python quick_update/imaging.py --n-repeats 5
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


import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402
from autonerves import jax_wrapper  # noqa: E402

sys.path.insert(0, str(_profiling_root()))

from instruments.imaging import INSTRUMENTS  # noqa: E402

try:
    from _profile_cli import device_info_dict
except ImportError:

    def device_info_dict():
        return {"backend": "unknown"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(prog="quick_update/imaging.py")
    p.add_argument("--instrument", default="hst", choices=list(INSTRUMENTS))
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--mask-radius", type=float, default=None)
    p.add_argument("--total-gaussians", type=int, default=20)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class Timer:
    def __init__(self):
        self.records: dict[str, list[float]] = {}

    @contextmanager
    def section(self, label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.setdefault(label, []).append(elapsed)
        print(f"  [{label}] {elapsed:.4f} s")

    def median(self, label: str) -> float:
        vals = sorted(self.records[label])
        n = len(vals)
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    def first(self, label: str) -> float:
        return self.records[label][0]


timer = Timer()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

args = _parse_args()
instrument = args.instrument
n_repeats = args.n_repeats
cfg = INSTRUMENTS[instrument]
mask_radius = args.mask_radius or cfg["mask_radius"]
total_gaussians = args.total_gaussians

workspace_root = _profiling_root()
dataset_path = workspace_root / "dataset" / "imaging" / instrument

print(f"Quick-update profiling: imaging MGE / {instrument}")
print(f"  dataset: {dataset_path}")
print(f"  repeats: {n_repeats}")
print(f"  MGE gaussians per galaxy: {total_gaussians}")
print()

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=cfg["pixel_scale"],
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

n_pixels = int(dataset.grid.shape[0])
print(f"  masked pixels: {n_pixels}")
print(f"  native shape:  {dataset.shape_native}")

# ---------------------------------------------------------------------------
# Model: MGE lens light + Isothermal mass + MGE source (start_here.py)
# ---------------------------------------------------------------------------

print(f"\n  Building MGE model ({total_gaussians} gaussians per galaxy)...")

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=total_gaussians,
    centre_prior_is_uniform=True,
)

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=lens_bulge,
    mass=mass,
    shear=shear,
)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=total_gaussians,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  free parameters: {model.prior_count}")

instance = model.instance_from_prior_medians()

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

tmp_dir = tempfile.mkdtemp(prefix="quick_update_profile_")
image_path = Path(tmp_dir)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

print("\n=== Warm-up (one-time JAX compilation) ===")

with timer.section("warmup"):
    fit = analysis.fit_for_visualization(instance=instance)
    _ = fit.model_data
    _ = fit.residual_map
    _ = fit.chi_squared_map

# ---------------------------------------------------------------------------
# Phase 1: fit_for_visualization breakdown
# ---------------------------------------------------------------------------

print("\n=== Phase 1: fit_for_visualization breakdown ===")

for i in range(n_repeats):
    inst = model.instance_from_prior_medians()

    with timer.section("fit_for_viz"):
        fit = analysis.fit_for_visualization(instance=inst)

    with timer.section("model_data"):
        _ = fit.model_data

    with timer.section("residual_map"):
        _ = fit.residual_map

    with timer.section("chi_squared_map"):
        _ = fit.chi_squared_map

# ---------------------------------------------------------------------------
# Phase 2: Critical curves
# ---------------------------------------------------------------------------

print("\n=== Phase 2: Critical curves ===")

from autolens.imaging.plot.fit_imaging_plots import (  # noqa: E402
    _compute_critical_curves_from_fit,
)

for i in range(n_repeats):
    with timer.section("critical_curves"):
        ip_lines, ip_colors, sp_lines, sp_colors = _compute_critical_curves_from_fit(fit)

# ---------------------------------------------------------------------------
# Phase 3: Render comparison (12-panel vs 6-panel)
# ---------------------------------------------------------------------------

print("\n=== Phase 3: subplot_fit_quick rendering (6-panel) ===")

from autolens.imaging.plot.fit_imaging_plots import subplot_fit_quick  # noqa: E402

render_kwargs = dict(
    output_path=str(image_path),
    output_format="png",
    image_plane_lines=ip_lines,
    image_plane_line_colors=ip_colors,
    source_plane_lines=sp_lines,
    source_plane_line_colors=sp_colors,
)

for i in range(n_repeats):
    with timer.section("subplot_fit_quick_render"):
        subplot_fit_quick(fit, **render_kwargs)

# ---------------------------------------------------------------------------
# Phase 4: End-to-end quick update
# ---------------------------------------------------------------------------

print("\n=== Phase 4: End-to-end quick update (6-panel) ===")

for i in range(n_repeats):
    inst = model.instance_from_prior_medians()
    with timer.section("end_to_end"):
        _fit = analysis.fit_for_visualization(instance=inst)
        _ = _fit.model_data
        _ = _fit.residual_map
        _ = _fit.chi_squared_map
        _ip, _ic, _sp, _sc = _compute_critical_curves_from_fit(_fit)
        subplot_fit_quick(
            _fit,
            output_path=str(image_path),
            output_format="png",
            image_plane_lines=_ip,
            image_plane_line_colors=_ic,
            source_plane_lines=_sp,
            source_plane_line_colors=_sc,
        )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print(f"SUMMARY — imaging MGE / {instrument} / {total_gaussians}g per galaxy")
print("=" * 65)

breakdown_phases = [
    ("fit_for_viz (construct FitImaging)", "fit_for_viz"),
    ("model_data (PSF conv + image)", "model_data"),
    ("residual_map", "residual_map"),
    ("chi_squared_map", "chi_squared_map"),
    ("critical_curves", "critical_curves"),
    ("subplot_fit_quick 6-panel (mpl)", "subplot_fit_quick_render"),
]

print(f"\n{'Phase':<38} | {'Median (s)':>10}")
print("-" * 65)

summary = {}
for label, key in breakdown_phases:
    t = timer.median(key)
    print(f"{label:<38} | {t:>10.4f}")
    summary[key] = round(t, 4)

print("-" * 65)
print(f"{'End-to-end (6-panel)':<38} | {timer.median('end_to_end'):>10.4f}")

print(f"\nWarmup (one-time): {timer.first('warmup'):.4f} s")

summary["end_to_end"] = round(timer.median("end_to_end"), 4)
summary["warmup"] = round(timer.first("warmup"), 4)

result = {
    "model": f"MGE {total_gaussians}g lens + {total_gaussians}g source",
    "instrument": instrument,
    "masked_pixels": n_pixels,
    "native_shape": list(dataset.shape_native),
    "mask_radius": mask_radius,
    "n_repeats": n_repeats,
    "device": device_info_dict(),
    "phases": summary,
    "all_timings": {k: [round(v, 4) for v in vals] for k, vals in timer.records.items()},
}

output_dir = (
    Path(args.output_dir) if args.output_dir else workspace_root / "results" / "quick_update"
)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"imaging_mge_quick_update_{instrument}.json"
output_path.write_text(json.dumps(result, indent=2))
print(f"\nResults written to {output_path}")
