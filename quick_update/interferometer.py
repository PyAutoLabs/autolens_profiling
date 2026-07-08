"""
Quick-Update Profiling: Interferometer MGE (SMA)
=================================================

Profiles the quick-update visualization path for an interferometer
model-fit with an **MGE source** (no lens light for interferometer).

Usage::

    cd autolens_profiling
    python quick_update/interferometer.py
    python quick_update/interferometer.py --instrument sma
    python quick_update/interferometer.py --n-repeats 5
"""

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
from autoconf import jax_wrapper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instruments.interferometer import INSTRUMENTS  # noqa: E402

try:
    from _profile_cli import device_info_dict
except ImportError:

    def device_info_dict():
        return {"backend": "unknown"}


import argparse  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(prog="quick_update/interferometer.py")
    p.add_argument("--instrument", default="sma", choices=list(INSTRUMENTS))
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--total-gaussians", type=int, default=20)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


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

args = _parse_args()
instrument = args.instrument
n_repeats = args.n_repeats
cfg = INSTRUMENTS[instrument]
total_gaussians = args.total_gaussians

workspace_root = Path(__file__).resolve().parents[1]
dataset_path = workspace_root / "dataset" / "interferometer" / instrument

print(f"Quick-update profiling: interferometer MGE / {instrument}")
print(f"  dataset: {dataset_path}")
print(f"  repeats: {n_repeats}")
print(f"  MGE source gaussians: {total_gaussians}")
print()

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=al.Mask2D.circular(
        shape_native=cfg["real_space_shape"],
        pixel_scales=cfg["pixel_scale"],
        radius=cfg["mask_radius"],
    ),
)

n_vis = int(dataset.data.shape[0])
print(f"  visibilities: {n_vis}")
print(f"  real-space shape: {cfg['real_space_shape']}")

# Model: Isothermal mass + shear + MGE source (no lens light)
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

source_bulge = al.model_util.mge_model_from(
    mask_radius=cfg["mask_radius"],
    total_gaussians=total_gaussians,
    centre_prior_is_uniform=False,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
print(f"  free parameters: {model.prior_count}")

instance = model.instance_from_prior_medians()
analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

tmp_dir = tempfile.mkdtemp(prefix="quick_update_interf_")
image_path = Path(tmp_dir)

# Warmup
print("\n=== Warm-up (one-time JAX compilation) ===")
with timer.section("warmup"):
    fit = analysis.fit_for_visualization(instance=instance)
    _ = fit.model_data
    _ = fit.dirty_image

# Phase 1: fit breakdown
print("\n=== Phase 1: fit_for_visualization breakdown ===")
for i in range(n_repeats):
    inst = model.instance_from_prior_medians()
    with timer.section("fit_for_viz"):
        fit = analysis.fit_for_visualization(instance=inst)
    with timer.section("model_data"):
        _ = fit.model_data
    with timer.section("dirty_images"):
        _ = fit.dirty_image
        _ = fit.dirty_model_image
        _ = fit.dirty_normalized_residual_map

# Phase 2: Critical curves
print("\n=== Phase 2: Critical curves ===")
from autolens.interferometer.plot.fit_interferometer_plots import (
    _compute_critical_curve_lines,  # noqa: E402
)

for i in range(n_repeats):
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    grid = fit.dataset.real_space_mask.derive_grid.all_false
    with timer.section("critical_curves"):
        _compute_critical_curve_lines(tracer, grid)

# Phase 3: Render
print("\n=== Phase 3: subplot_fit_quick rendering (6-panel) ===")
from autolens.interferometer.plot.fit_interferometer_plots import subplot_fit_quick  # noqa: E402

subplot_fit_quick(fit, output_path=str(image_path), output_format="png")

for i in range(n_repeats):
    with timer.section("subplot_fit_quick_render"):
        subplot_fit_quick(fit, output_path=str(image_path), output_format="png")

# Phase 4: End-to-end
print("\n=== Phase 4: End-to-end quick update (6-panel) ===")
for i in range(n_repeats):
    inst = model.instance_from_prior_medians()
    with timer.section("end_to_end"):
        _fit = analysis.fit_for_visualization(instance=inst)
        _ = _fit.model_data
        subplot_fit_quick(_fit, output_path=str(image_path), output_format="png")

# Summary
print("\n" + "=" * 65)
print(f"SUMMARY — interferometer MGE / {instrument}")
print("=" * 65)

phases = [
    ("fit_for_viz", "fit_for_viz"),
    ("model_data", "model_data"),
    ("dirty_images (3x FFT)", "dirty_images"),
    ("critical_curves", "critical_curves"),
    ("subplot_fit_quick 6-panel", "subplot_fit_quick_render"),
]

print(f"\n{'Phase':<38} | {'Median (s)':>10}")
print("-" * 65)

summary = {}
for label, key in phases:
    t = timer.median(key)
    print(f"{label:<38} | {t:>10.4f}")
    summary[key] = round(t, 4)

print("-" * 65)
print(f"{'End-to-end (6-panel)':<38} | {timer.median('end_to_end'):>10.4f}")
print(f"\nWarmup (one-time): {timer.first('warmup'):.4f} s")

summary["end_to_end"] = round(timer.median("end_to_end"), 4)
summary["warmup"] = round(timer.first("warmup"), 4)

result = {
    "model": f"MGE {total_gaussians}g source (no lens light)",
    "instrument": instrument,
    "n_visibilities": n_vis,
    "real_space_shape": list(cfg["real_space_shape"]),
    "n_repeats": n_repeats,
    "device": device_info_dict(),
    "phases": summary,
    "all_timings": {k: [round(v, 4) for v in vals] for k, vals in timer.records.items()},
}

output_dir = (
    Path(args.output_dir) if args.output_dir else workspace_root / "results" / "quick_update"
)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"interferometer_mge_quick_update_{instrument}.json"
output_path.write_text(json.dumps(result, indent=2))
print(f"\nResults written to {output_path}")
