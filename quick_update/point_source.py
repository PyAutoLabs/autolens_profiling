"""
Quick-Update Profiling: Point Source
====================================

Profiles the quick-update visualization path for a point-source
model-fit. Point source fits are much cheaper than imaging/interferometer
(no pixelized images to render), so this is primarily a correctness
check that the subplot_fit_quick pipeline works.

Usage::

    cd autolens_profiling
    python quick_update/point_source.py
"""

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from autoconf import jax_wrapper  # noqa: E402
from autoconf.dictable import from_dict  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from _profile_cli import device_info_dict
except ImportError:
    def device_info_dict():
        return {"backend": "unknown"}


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
n_repeats = 3

workspace_root = Path(__file__).resolve().parents[1]
dataset_path = workspace_root / "dataset" / "point_source" / "simple"

print("Quick-update profiling: point source")
print(f"  dataset: {dataset_path}")
print(f"  repeats: {n_repeats}")
print()

# Load dataset
point_dataset = from_dict(
    json.loads((dataset_path / "point_dataset_positions_only.json").read_text())
)
print(f"  positions: {len(point_dataset.positions)} images")

# Load tracer for model construction
tracer = from_dict(json.loads((dataset_path / "tracer.json").read_text()))

# Build model
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
source = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.Point())
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  free parameters: {model.prior_count}")

instance = model.instance_from_prior_medians()
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05),
    pixel_scale_precision=0.001,
)
analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver, use_jax=True)

tmp_dir = tempfile.mkdtemp(prefix="quick_update_point_")

# Warmup
print("\n=== Warm-up ===")
with timer.section("warmup"):
    fit = analysis.fit_for_visualization(instance=instance)

# Phase 1: fit breakdown
print("\n=== Phase 1: fit_for_visualization ===")
for i in range(n_repeats):
    inst = model.instance_from_prior_medians()
    with timer.section("fit_for_viz"):
        fit = analysis.fit_for_visualization(instance=inst)

# Phase 2: Render
print("\n=== Phase 2: subplot_fit_quick rendering ===")
from autolens.point.plot.fit_point_plots import subplot_fit_quick  # noqa: E402

subplot_fit_quick(fit, output_path=tmp_dir, output_format="png")

for i in range(n_repeats):
    with timer.section("subplot_fit_quick_render"):
        subplot_fit_quick(fit, output_path=tmp_dir, output_format="png")

# Phase 3: End-to-end
print("\n=== Phase 3: End-to-end quick update ===")
for i in range(n_repeats):
    inst = model.instance_from_prior_medians()
    with timer.section("end_to_end"):
        _fit = analysis.fit_for_visualization(instance=inst)
        subplot_fit_quick(_fit, output_path=tmp_dir, output_format="png")

# Summary
print("\n" + "=" * 55)
print("SUMMARY — point source")
print("=" * 55)

phases = [
    ("fit_for_viz", "fit_for_viz"),
    ("subplot_fit_quick render", "subplot_fit_quick_render"),
]

print(f"\n{'Phase':<30} | {'Median (s)':>10}")
print("-" * 55)

summary = {}
for label, key in phases:
    t = timer.median(key)
    print(f"{label:<30} | {t:>10.4f}")
    summary[key] = round(t, 4)

print("-" * 55)
print(f"{'End-to-end':<30} | {timer.median('end_to_end'):>10.4f}")
print(f"\nWarmup (one-time): {timer.first('warmup'):.4f} s")

summary["end_to_end"] = round(timer.median("end_to_end"), 4)
summary["warmup"] = round(timer.first("warmup"), 4)

result = {
    "model": "Isothermal + shear + Point source",
    "n_positions": len(point_dataset.positions),
    "n_repeats": n_repeats,
    "device": device_info_dict(),
    "phases": summary,
    "all_timings": {
        k: [round(v, 4) for v in vals]
        for k, vals in timer.records.items()
    },
}

output_dir = workspace_root / "results" / "quick_update"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "point_source_quick_update.json"
output_path.write_text(json.dumps(result, indent=2))
print(f"\nResults written to {output_path}")
