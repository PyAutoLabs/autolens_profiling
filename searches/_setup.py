"""
Shared setup for the ``searches/`` profiling scripts.

Builds the HST imaging dataset, the MGE + Isothermal + ExternalShear lens model
with an MGE source bulge, and the ``AnalysisImaging`` object used by every
sampler in this section. The dataset, mask, and model mirror the reference setup
in ``likelihood/imaging/mge.py`` so the likelihood value is directly comparable
across the two sections.

Usage
-----

    from searches._setup import build_dataset, build_model, build_analysis

    dataset = build_dataset()
    model = build_model(mask_radius=3.5)
    analysis = build_analysis(dataset, use_jax=False)
"""

from pathlib import Path

import numpy as np

import autofit as af
import autolens as al

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_DATASET_SUBPATH = Path("dataset") / "imaging" / "hst"

PIXEL_SCALE = 0.05
MASK_RADIUS = 3.5


def build_dataset(mask_radius: float = MASK_RADIUS) -> al.Imaging:
    """Load the HST imaging dataset with mask + radial-bin over-sampling applied."""
    dataset_path = _DATASET_SUBPATH

    if al.util.dataset.should_simulate(str(dataset_path)):
        raise FileNotFoundError(
            f"Input dataset missing at '{dataset_path}'. The autolens_profiling "
            f"repo mirrors only the curated datasets needed for default smoke "
            f"runs. To regenerate, use the source-of-truth script at "
            f"autolens_workspace_developer/jax_profiling/dataset_setup/imaging.py "
            f"and copy the result into autolens_profiling/dataset/."
        )

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=PIXEL_SCALE,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
    return dataset


def build_model(
    mask_radius: float = MASK_RADIUS, total_gaussians: int = 20
) -> af.Collection:
    """Build the lens + source model used in ``jax_profiling/imaging/mge.py``."""
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=total_gaussians,
        centre_prior_is_uniform=True,
    )
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=total_gaussians,
        centre_prior_is_uniform=False,
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def build_analysis(dataset: al.Imaging, use_jax: bool = False) -> al.AnalysisImaging:
    """Build the analysis object. Set ``use_jax=True`` for the pure-JAX path."""
    return al.AnalysisImaging(dataset=dataset, use_jax=use_jax)


def format_best_fit(instance) -> str:
    """Terse one-line summary of the lens mass + shear of a best-fit instance."""
    mass = instance.galaxies.lens.mass
    shear = instance.galaxies.lens.shear
    return (
        f"lens.mass.einstein_radius={mass.einstein_radius:.4f}  "
        f"lens.mass.centre=({mass.centre[0]:.3f}, {mass.centre[1]:.3f})  "
        f"shear=({shear.gamma_1:.4f}, {shear.gamma_2:.4f})"
    )
