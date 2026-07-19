"""
Quick-Update Profiling: Imaging Delaunay (HST)
==============================================

Profiles the quick-update visualization path for an imaging model-fit
with an **MGE lens + Delaunay pixelized source** — the production-grade
inversion model.

The Delaunay source uses a Hilbert image-mesh (1500 vertices) with
ConstantSplit regularization. The lens light is a 20-Gaussian MGE.

This is the counterpart to ``imaging.py`` (MGE lens + MGE source).
Together they cover the two main model types users run.

Usage::

    cd autolens_profiling
    python quick_update/imaging_delaunay.py
    python quick_update/imaging_delaunay.py --instrument euclid
    python quick_update/imaging_delaunay.py --n-repeats 5
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
    p = argparse.ArgumentParser(prog="quick_update/imaging_delaunay.py")
    p.add_argument("--instrument", default="hst", choices=list(INSTRUMENTS))
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--mask-radius", type=float, default=None)
    p.add_argument("--lens-gaussians", type=int, default=20)
    p.add_argument("--mesh-pixels", type=int, default=1500)
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
lens_gaussians = args.lens_gaussians
mesh_pixels = args.mesh_pixels

workspace_root = Path(__file__).resolve().parents[1]
dataset_path = workspace_root / "dataset" / "imaging" / instrument

print(f"Quick-update profiling: imaging Delaunay / {instrument}")
print(f"  dataset: {dataset_path}")
print(f"  repeats: {n_repeats}")
print(f"  lens MGE gaussians: {lens_gaussians}")
print(f"  Delaunay mesh pixels: {mesh_pixels}")
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
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)

n_pixels = int(dataset.grid.shape[0])
print(f"  masked pixels: {n_pixels}")
print(f"  native shape:  {dataset.shape_native}")

# ---------------------------------------------------------------------------
# Adapt image (needed for Hilbert image-mesh)
# ---------------------------------------------------------------------------

print("\n  Loading adapt image...")
adapt_image = adapt_image_for_dataset(
    dataset_path=dataset_path,
    dataset=dataset,
)
print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

# ---------------------------------------------------------------------------
# Model: MGE lens light + Isothermal mass + Delaunay source
# ---------------------------------------------------------------------------

print("\n  Building Delaunay model...")

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=lens_gaussians,
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

mesh = al.mesh.Delaunay(pixels=mesh_pixels, zeroed_pixels=0)
regularization = al.reg.ConstantSplit(coefficient=1.0)
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  free parameters: {model.prior_count}")

instance = model.instance_from_prior_medians()

print("\n  Building Hilbert image-plane mesh grid...")
image_mesh = al.image_mesh.Hilbert(
    pixels=mesh_pixels,
    weight_power=1.0,
    weight_floor=0.0,
)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
    adapt_data=adapt_image,
)
print(f"  mesh vertices: {image_plane_mesh_grid.shape[0]}")

source_key = str(("galaxies", "source"))

adapt_images = al.AdaptImages(
    galaxy_name_image_dict={source_key: adapt_image},
    galaxy_name_image_plane_mesh_grid_dict={
        source_key: image_plane_mesh_grid,
    },
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
analysis.adapt_images = adapt_images

tmp_dir = tempfile.mkdtemp(prefix="quick_update_profile_delaunay_")
image_path = Path(tmp_dir)

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
# Phase 3: Render comparison
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
print(f"SUMMARY — imaging Delaunay / {instrument}")
print(f"  lens: MGE {lens_gaussians}g | source: Delaunay {mesh_pixels}px")
print("=" * 65)

breakdown_phases = [
    ("fit_for_viz (construct FitImaging)", "fit_for_viz"),
    ("model_data (PSF conv + inversion)", "model_data"),
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
    "model": f"MGE {lens_gaussians}g lens + Delaunay {mesh_pixels}px source",
    "instrument": instrument,
    "masked_pixels": n_pixels,
    "native_shape": list(dataset.shape_native),
    "mask_radius": mask_radius,
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
output_path = output_dir / f"imaging_delaunay_quick_update_{instrument}.json"
output_path.write_text(json.dumps(result, indent=2))
print(f"\nResults written to {output_path}")
