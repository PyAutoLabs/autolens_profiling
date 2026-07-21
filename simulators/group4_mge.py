"""
Simulator: 4-Lens + 4-Source Group-Scale Imaging (MGE search benchmark)
========================================================================

Simulates a **group-scale** imaging dataset with **4 deflector galaxies** and
**4 background source galaxies** — the higher-dimensional, harder-to-fit model
that ``searches/<sampler>/group/mge.py`` benchmarks the JAX gradient MAP
optimizers against ``af.Nautilus`` on (autolens_profiling#82).

Unlike ``simulators/imaging.py`` (single lens + single source) this writes an
extra **``truth.json``** capturing the input mass + geometry of every galaxy, so
the search harness can score whether a fit actually *recovers the input truth*
(``searches/_recovery.py``).

The truth geometry is defined once in ``GROUP4_TRUTH`` and used to build both the
``al.Tracer`` (for simulation) and ``truth.json`` (for recovery scoring) — a
single source of truth. Deflectors are kept compact (within ~1.8") so the
standard per-instrument ``mask_radius`` (3.5") contains the whole group and the
per-evaluation cost stays comparable to the single-lens MGE cell.

Sources are simulated with parametric ``al.lp.SersicCore`` light; the search
*fits* them with MGE bases (linear light) — mirroring the single-lens cell,
where an MGE basis approximates a parametric truth. MGE needs no adapt image and
the benchmark needs no image-plane positions, so (unlike ``imaging.py``) this
simulator writes neither ``lensed_source.fits`` nor ``positions.json``.

Usage
-----

    python simulators/group4_mge.py                    # hst (default)
    python simulators/group4_mge.py --instrument euclid
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instruments.imaging import INSTRUMENTS  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]  # autolens_profiling/

# ---------------------------------------------------------------------------
# Truth geometry — the single source of truth for both the simulated tracer and
# the ``truth.json`` recovery target. Angles are degrees; positions are arcsec
# in the (y, x) autolens convention.
# ---------------------------------------------------------------------------

GROUP4_TRUTH: dict = {
    "redshift_lens": 0.5,
    "redshift_source": 1.0,
    # 4 deflectors: Isothermal mass (+ Sersic light); shear on the primary only.
    "lenses": [
        {
            "name": "lens_0",
            "centre": (0.0, 0.0),
            "einstein_radius": 1.2,
            "axis_ratio": 0.9,
            "angle": 45.0,
            "light_intensity": 2.0,
            "light_effective_radius": 0.6,
            "light_sersic_index": 3.0,
            "shear": (0.03, 0.04),  # (gamma_1, gamma_2); None on the others
        },
        {
            "name": "lens_1",
            "centre": (1.6, 1.1),
            "einstein_radius": 0.50,
            "axis_ratio": 0.85,
            "angle": 30.0,
            "light_intensity": 1.0,
            "light_effective_radius": 0.4,
            "light_sersic_index": 3.0,
            "shear": None,
        },
        {
            "name": "lens_2",
            "centre": (-1.4, 1.3),
            "einstein_radius": 0.45,
            "axis_ratio": 0.80,
            "angle": 100.0,
            "light_intensity": 1.0,
            "light_effective_radius": 0.4,
            "light_sersic_index": 3.0,
            "shear": None,
        },
        {
            "name": "lens_3",
            "centre": (0.6, -1.7),
            "einstein_radius": 0.50,
            "axis_ratio": 0.90,
            "angle": 70.0,
            "light_intensity": 1.0,
            "light_effective_radius": 0.4,
            "light_sersic_index": 3.0,
            "shear": None,
        },
    ],
    # 4 sources at distinct source-plane positions so each is separately lensed.
    "sources": [
        {
            "name": "source_0",
            "centre": (0.05, 0.0),
            "axis_ratio": 0.8,
            "angle": 60.0,
            "intensity": 4.0,
            "effective_radius": 0.10,
            "sersic_index": 1.0,
        },
        {
            "name": "source_1",
            "centre": (0.40, 0.30),
            "axis_ratio": 0.7,
            "angle": 20.0,
            "intensity": 3.0,
            "effective_radius": 0.08,
            "sersic_index": 1.5,
        },
        {
            "name": "source_2",
            "centre": (-0.30, 0.25),
            "axis_ratio": 0.85,
            "angle": 100.0,
            "intensity": 3.0,
            "effective_radius": 0.09,
            "sersic_index": 1.0,
        },
        {
            "name": "source_3",
            "centre": (0.25, -0.35),
            "axis_ratio": 0.9,
            "angle": 140.0,
            "intensity": 2.5,
            "effective_radius": 0.08,
            "sersic_index": 1.2,
        },
    ],
}


def _build_tracer(al):
    """Construct the 4-lens + 4-source ``al.Tracer`` from ``GROUP4_TRUTH``."""
    galaxies = []
    for lens in GROUP4_TRUTH["lenses"]:
        kwargs = dict(
            redshift=GROUP4_TRUTH["redshift_lens"],
            bulge=al.lp.Sersic(
                centre=lens["centre"],
                ell_comps=al.convert.ell_comps_from(
                    axis_ratio=lens["axis_ratio"], angle=lens["angle"]
                ),
                intensity=lens["light_intensity"],
                effective_radius=lens["light_effective_radius"],
                sersic_index=lens["light_sersic_index"],
            ),
            mass=al.mp.Isothermal(
                centre=lens["centre"],
                einstein_radius=lens["einstein_radius"],
                ell_comps=al.convert.ell_comps_from(
                    axis_ratio=lens["axis_ratio"], angle=lens["angle"]
                ),
            ),
        )
        if lens["shear"] is not None:
            kwargs["shear"] = al.mp.ExternalShear(
                gamma_1=lens["shear"][0], gamma_2=lens["shear"][1]
            )
        galaxies.append(al.Galaxy(**kwargs))

    for source in GROUP4_TRUTH["sources"]:
        galaxies.append(
            al.Galaxy(
                redshift=GROUP4_TRUTH["redshift_source"],
                bulge=al.lp.SersicCore(
                    centre=source["centre"],
                    ell_comps=al.convert.ell_comps_from(
                        axis_ratio=source["axis_ratio"], angle=source["angle"]
                    ),
                    intensity=source["intensity"],
                    effective_radius=source["effective_radius"],
                    sersic_index=source["sersic_index"],
                ),
            )
        )
    return al.Tracer(galaxies=galaxies)


def simulate(instrument: str = "hst", output_root: Path | None = None) -> Path:
    """Simulate the 4-lens + 4-source group dataset. Returns the dataset dir."""
    import json
    import time
    from contextlib import contextmanager

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import autolens as al
    import autolens.plot as aplt
    import matplotlib.pyplot as plt

    if instrument not in INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument '{instrument}'. Choose from: {list(INSTRUMENTS.keys())}"
        )

    config = INSTRUMENTS[instrument]
    pixel_scale = config["pixel_scale"]
    mask_radius = config["mask_radius"]
    psf_shape = config["psf_shape"]
    psf_sigma = config["psf_sigma"]
    seed = config["seed"]

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "imaging" / "group4_mge" / instrument
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Grid sized so the circular mask_radius mask fits (tight odd bounding box).
    shape_pixels = int(np.ceil(2 * mask_radius / pixel_scale))
    if shape_pixels % 2 == 0:
        shape_pixels += 1

    records: list[tuple[str, float]] = []

    @contextmanager
    def section(label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    print(f"\n--- Group4 MGE simulator [{instrument}] ---")
    print(f"  pixel_scale:     {pixel_scale} arcsec/px")
    print(f"  grid_shape:      {shape_pixels} x {shape_pixels}")
    print(f"  mask_radius:     {mask_radius} arcsec")
    print(f"  n_lenses:        {len(GROUP4_TRUTH['lenses'])}")
    print(f"  n_sources:       {len(GROUP4_TRUTH['sources'])}")
    print(f"  output:          {dataset_path}")

    # Over-sample densely at every deflector centre (each hosts a cuspy light +
    # mass profile) so the simulated light profiles are integrated accurately.
    lens_centres = [tuple(map(float, ln["centre"])) for ln in GROUP4_TRUTH["lenses"]]

    with section("setup_grids"):
        grid = al.Grid2D.uniform(
            shape_native=(shape_pixels, shape_pixels), pixel_scales=pixel_scale
        )
        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.3, 0.6],
            centre_list=lens_centres,
        )
        grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

    with section("setup_psf_simulator"):
        psf = al.Convolver.from_gaussian(
            shape_native=psf_shape, sigma=psf_sigma, pixel_scales=grid.pixel_scales
        )
        simulator = al.SimulatorImaging(
            exposure_time=300.0,
            psf=psf,
            background_sky_level=0.1,
            add_poisson_noise_to_data=True,
            noise_seed=seed,
        )

    with section("setup_tracer"):
        tracer = _build_tracer(al)

    with section("via_tracer_from"):
        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    # === outputs ===

    print("\n--- outputs ---")

    with section("output_fits"):
        aplt.fits_imaging(
            dataset=dataset,
            data_path=dataset_path / "data.fits",
            psf_path=dataset_path / "psf.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            overwrite=True,
        )

    with section("output_json"):
        al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
        (dataset_path / "truth.json").write_text(json.dumps(_truth_record(), indent=2))

    with section("output_preview_png"):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.asarray(dataset.data.native), origin="lower", cmap="inferno")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"group4_mge [{instrument}] — 4 lenses + 4 sources")
        fig.tight_layout()
        fig.savefig(dataset_path / "data_preview.png", dpi=150)
        plt.close(fig)

    # === summary ===

    al_version = al.__version__
    results_dir = root / "results" / "simulators"
    results_dir.mkdir(parents=True, exist_ok=True)
    phases = dict(records)

    results_summary = {
        "autolens_version": al_version,
        "type": "group4_mge",
        "instrument": instrument,
        "configuration": {
            "grid_shape": [shape_pixels, shape_pixels],
            "pixel_scales": pixel_scale,
            "mask_radius": mask_radius,
            "psf_shape": list(psf_shape),
            "psf_sigma": psf_sigma,
            "n_lenses": len(GROUP4_TRUTH["lenses"]),
            "n_sources": len(GROUP4_TRUTH["sources"]),
        },
        "phases": phases,
    }
    json_path = results_dir / f"group4_mge_{instrument}_summary_v{al_version}.json"
    json_path.write_text(json.dumps(results_summary, indent=2))
    print(f"\n  Results saved to: {json_path}")
    print(f"  Dataset + truth.json written to: {dataset_path}")

    return dataset_path


def _truth_record() -> dict:
    """Recovery-scoring truth: the geometry ``searches/_recovery.py`` compares
    a fit's ``max_log_likelihood_instance`` against.

    MGE amplitudes are linear nuisance parameters (solved by the inversion), so
    the recoverable quantities are the mass parameters + light/mass centres, not
    the light intensities — those are omitted from the scoring target.
    """
    return {
        "lenses": [
            {
                "name": ln["name"],
                "centre": list(map(float, ln["centre"])),
                "einstein_radius": float(ln["einstein_radius"]),
                "shear": None if ln["shear"] is None else list(map(float, ln["shear"])),
            }
            for ln in GROUP4_TRUTH["lenses"]
        ],
        "sources": [
            {"name": src["name"], "centre": list(map(float, src["centre"]))}
            for src in GROUP4_TRUTH["sources"]
        ],
    }


if __name__ == "__main__":
    import argparse
    import os

    from autonerves import jax_wrapper  # noqa: F401

    if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
        print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
        sys.exit(0)

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--instrument",
        type=str,
        default="hst",
        choices=list(INSTRUMENTS.keys()),
        help="Instrument preset to simulate (default: hst).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the autolens_profiling root that holds dataset/.",
    )
    args = parser.parse_args()
    simulate(instrument=args.instrument, output_root=args.output_root)
