"""
Quick-Update Profiling: Interferometer Delaunay (SMA)
=====================================================

Profiles the quick-update visualization path for an interferometer
model-fit with a **Delaunay pixelized source** (Hilbert image-mesh).

Usage::

    cd autolens_profiling
    python quick_update/interferometer_delaunay.py
    python quick_update/interferometer_delaunay.py --instrument sma
"""

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402
from autonerves import jax_wrapper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from instruments.interferometer import INSTRUMENTS  # noqa: E402

try:
    from _profile_cli import device_info_dict
except ImportError:

    def device_info_dict():
        return {"backend": "unknown"}


import argparse  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(prog="quick_update/interferometer_delaunay.py")
    p.add_argument("--instrument", default="sma", choices=list(INSTRUMENTS))
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--mesh-pixels", type=int, default=1500)
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
mesh_pixels = args.mesh_pixels

workspace_root = Path(__file__).resolve().parents[1]
dataset_path = workspace_root / "dataset" / "interferometer" / instrument

print(f"Quick-update profiling: interferometer Delaunay / {instrument}")
print(f"  dataset: {dataset_path}")
print(f"  repeats: {n_repeats}")
print(f"  Delaunay mesh pixels: {mesh_pixels}")
print()

real_space_mask = al.Mask2D.circular(
    shape_native=cfg["real_space_shape"],
    pixel_scales=cfg["pixel_scale"],
    radius=cfg["mask_radius"],
)

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
)

n_vis = int(dataset.data.shape[0])
print(f"  visibilities: {n_vis}")
print(f"  real-space shape: {cfg['real_space_shape']}")

# Adapt image
print("\n  Loading adapt image...")
adapt_image = adapt_image_for_dataset(
    dataset_path=dataset_path,
    dataset=dataset,
)
print(f"  adapt_image shape: {adapt_image.shape_slim}")

print("  Building Hilbert mesh grid...")
image_mesh = al.image_mesh.Hilbert(pixels=mesh_pixels, weight_power=1.0, weight_floor=0.0)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=real_space_mask,
    adapt_data=adapt_image,
)
print(f"  mesh vertices: {image_plane_mesh_grid.shape[0]}")

# Model: Isothermal + shear + Delaunay source
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

mesh = al.mesh.Delaunay(pixels=mesh_pixels, zeroed_pixels=0)
regularization = al.reg.ConstantSplit(coefficient=1.0)
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
print(f"  free parameters: {model.prior_count}")

instance = model.instance_from_prior_medians()

source_key = str(("galaxies", "source"))
adapt_images = al.AdaptImages(
    galaxy_name_image_dict={source_key: adapt_image},
    galaxy_name_image_plane_mesh_grid_dict={source_key: image_plane_mesh_grid},
)

analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)
analysis.adapt_images = adapt_images

tmp_dir = tempfile.mkdtemp(prefix="quick_update_interf_del_")
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

# Phase 2: Render
print("\n=== Phase 2: subplot_fit_quick rendering (6-panel) ===")
from autolens.interferometer.plot.fit_interferometer_plots import subplot_fit_quick  # noqa: E402

subplot_fit_quick(fit, output_path=str(image_path), output_format="png")

for i in range(n_repeats):
    with timer.section("subplot_fit_quick_render"):
        subplot_fit_quick(fit, output_path=str(image_path), output_format="png")

# Phase 3: End-to-end
print("\n=== Phase 3: End-to-end quick update (6-panel) ===")
for i in range(n_repeats):
    inst = model.instance_from_prior_medians()
    with timer.section("end_to_end"):
        _fit = analysis.fit_for_visualization(instance=inst)
        _ = _fit.model_data
        subplot_fit_quick(_fit, output_path=str(image_path), output_format="png")

# Summary
print("\n" + "=" * 65)
print(f"SUMMARY — interferometer Delaunay / {instrument}")
print(f"  source: Delaunay {mesh_pixels}px")
print("=" * 65)

phases = [
    ("fit_for_viz", "fit_for_viz"),
    ("model_data (visibilities)", "model_data"),
    ("dirty_images (3x FFT)", "dirty_images"),
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
    "model": f"Delaunay {mesh_pixels}px source",
    "instrument": instrument,
    "n_visibilities": n_vis,
    "real_space_shape": list(cfg["real_space_shape"]),
    "mesh_pixels": mesh_pixels,
    "n_repeats": n_repeats,
    "device": device_info_dict(),
    "phases": summary,
    "all_timings": {k: [round(v, 4) for v in vals] for k, vals in timer.records.items()},
}

output_dir = (
    Path(args.output_dir) if args.output_dir else workspace_root / "results" / "quick_update"
)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"interferometer_delaunay_quick_update_{instrument}.json"
output_path.write_text(json.dumps(result, indent=2))
print(f"\nResults written to {output_path}")
