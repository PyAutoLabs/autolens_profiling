"""The multi_galaxy and group standard cells (likelihood-lifecycle 2d, autolens_profiling#473).

One definition of each dataset, its masked and over-sampled form, and the cell model, so every
setup of a likelihood (JAX dense / sparse, numba sparse; runtime and breakdown) fits the same
thing. Nothing runs on import (stdlib only at module level); callers own timing and output.

Datasets mirror the workspace simulators and the autolens_visualization producers
(``autolens_workspace/scripts/{multi_galaxy,group}/simulator.py``,
``autolens_visualization/true_models/{multi_galaxy,group}.py``): the same galaxies, grid,
exposure, sky and over-sampling. The PSF is the instrument preset's
(``instruments/imaging.py``, 21 x 21), so the cells compare with the imaging cells of the same
instrument (decision D2 on the campaign page).

- ``multi_galaxy`` at ``hst`` (0.05"/px, 200 x 200, 3.0" mask): two co-dominant lenses
  (Sersic light + Isothermal mass), an external shear, a light-only extra galaxy whose pixels are
  noise-scaled (``mask_extra_galaxies``) as ``multi_galaxy/modeling.py`` does, and a SersicCore
  source.
- ``group`` at ``euclid`` (0.1"/px, 250 x 250, 7.5" mask): one main lens (SersicSph +
  IsothermalSph), two extra galaxies (SersicSph + dPIEMassSph), a SersicCore source.

The cell model is the workspace modeling model (``{multi_galaxy,group}/modeling.py``, the MGE
form): each main lens an MGE of 20 linear Gaussians + Isothermal, an external shear, each group
extra galaxy an MGE of 10 Gaussians (centre fixed) + dPIEMassSph (sigma free), and the source an
MGE of 20 Gaussians (variant ``mge``) or the standard-size production pixelization (variants
``delaunay`` / ``rectangular``: the hst production preset at the standard size, as the imaging
cells, ``_production_config.standard_size_preset``). Mass priors are narrow Gaussians on the
truth so the seeded iid draws (central 20 % of every prior, the imaging cells' sequence) stay
physical.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

#: The dataset types and likelihood variants this module builds.
DATASET_TYPES = ("multi_galaxy", "group")
VARIANTS = ("mge", "rectangular", "delaunay")
PIXELIZED = ("rectangular", "delaunay")


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset type's simulation and masking (the workspace simulator, instrument PSF)."""

    dataset_type: str
    instrument: str
    pixel_scale: float
    shape_native: tuple[int, int]
    mask_radius: float
    psf_shape: tuple[int, int]
    psf_sigma: float
    exposure_time: float
    background_sky_level: float
    noise_seed: int
    main_lens_centres: tuple
    extra_galaxies_centres: tuple = ()
    #: (centre, radius) of the noise-scaled extra-galaxy region (multi_galaxy only).
    noise_scaled: tuple | None = None
    provenance: str = ""
    lp_sub_size_list: tuple = (4, 2, 2)
    lp_radial_list: tuple = (0.3, 0.6)
    simulate_sub_size_list: tuple = (32, 8, 2)
    notes: list = field(default_factory=list)

    @property
    def centres(self) -> list[tuple[float, float]]:
        """Every galaxy centre the light-profile over-sampling is centred on (workspace rule)."""
        return [tuple(c) for c in (*self.main_lens_centres, *self.extra_galaxies_centres)]

    @property
    def masked_pixels(self) -> int:
        """Image pixels inside the circular mask: pi r^2 / pixel_scale^2."""
        return int(math.pi * self.mask_radius**2 / self.pixel_scale**2)


SPECS = {
    "multi_galaxy": DatasetSpec(
        dataset_type="multi_galaxy",
        instrument="hst",
        pixel_scale=0.05,
        shape_native=(200, 200),
        mask_radius=3.0,
        psf_shape=(21, 21),
        psf_sigma=0.05,
        exposure_time=900.0,
        background_sky_level=0.1,
        noise_seed=1,
        main_lens_centres=((0.35, 0.25), (-0.35, -0.25)),
        noise_scaled=((2.2, 1.6), 0.9),
        provenance=(
            'autolens_workspace scripts/multi_galaxy/simulator.py (grid 200x200 at 0.05", '
            'galaxies, exposure 900 s, sky 0.1, mask_extra_galaxies 3 x 0.3") and '
            'modeling.py (3.0" mask, noise scaling, radial-bin over-sampling on both lenses); '
            "autolens_visualization true_models/multi_galaxy.py; PSF from instruments/imaging.py "
            'hst (21x21, sigma 0.05")'
        ),
    ),
    "group": DatasetSpec(
        dataset_type="group",
        instrument="euclid",
        pixel_scale=0.1,
        shape_native=(250, 250),
        mask_radius=7.5,
        psf_shape=(21, 21),
        psf_sigma=0.1,
        exposure_time=300.0,
        background_sky_level=0.1,
        noise_seed=1,
        main_lens_centres=((0.0, 0.0),),
        extra_galaxies_centres=((3.5, 2.5), (-4.4, -5.0)),
        provenance=(
            'autolens_workspace scripts/group/simulator.py (grid 250x250 at 0.1", galaxies, '
            'exposure 300 s, sky 0.1) and modeling.py (7.5" mask, radial-bin over-sampling on '
            "every galaxy); autolens_visualization true_models/group.py; PSF from "
            'instruments/imaging.py euclid (21x21, sigma 0.1")'
        ),
    ),
}

#: The instrument each dataset type is measured on (decision D2).
INSTRUMENTS = {dt: spec.instrument for dt, spec in SPECS.items()}

#: The seeded iid instance sequence (the imaging production presets': unit values uniform in the
#: central 20 % of every prior, seed 235), used by the MGE variant, which has no mesh preset.
IID_SEED = 235
IID_UNIT_LOW, IID_UNIT_HIGH = 0.4, 0.6


def spec_for(dataset_type: str, instrument: str | None = None) -> DatasetSpec:
    if dataset_type not in SPECS:
        raise KeyError(f"no galaxy/group cell for dataset type {dataset_type!r}")
    spec = SPECS[dataset_type]
    if instrument not in (None, spec.instrument):
        raise SystemExit(
            f"{dataset_type} is planned at {spec.instrument} only (decision D2); got {instrument!r}"
        )
    return spec


def dataset_path(dataset_type: str, root: Path) -> Path:
    spec = SPECS[dataset_type]
    return Path(root) / "dataset" / dataset_type / spec.instrument


# --------------------------------------------------------------------------- truth galaxies


def truth_galaxies(dataset_type: str):
    """``(galaxies, fields)`` of the simulator's true model (the visualization true models)."""
    import autolens as al

    if dataset_type == "multi_galaxy":
        lens_0 = al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(0.35, 0.25),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
                intensity=1.2,
                effective_radius=0.6,
                sersic_index=4.0,
            ),
            mass=al.mp.Isothermal(
                centre=(0.30, 0.28),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
                einstein_radius=1.0,
            ),
        )
        lens_1 = al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(-0.35, -0.25),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
                intensity=1.0,
                effective_radius=0.5,
                sersic_index=4.0,
            ),
            mass=al.mp.Isothermal(
                centre=(-0.31, -0.22),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
                einstein_radius=0.8,
            ),
        )
        extra = al.Galaxy(
            redshift=0.5,
            light=al.lp.ExponentialSph(centre=(2.2, 1.6), intensity=1.0, effective_radius=0.3),
        )
        source = al.Galaxy(
            redshift=1.0,
            bulge=al.lp.SersicCore(
                centre=(0.0, 0.03),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                intensity=3.0,
                effective_radius=0.15,
                sersic_index=1.0,
            ),
        )
        field = al.MassField(redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05))
        return [lens_0, lens_1, extra, source], [field]

    if dataset_type == "group":
        lens_0 = al.Galaxy(
            redshift=0.5,
            bulge=al.lp.SersicSph(
                centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
            ),
            mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
        )
        extras = [
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.SersicSph(
                    centre=centre, intensity=0.9, effective_radius=0.8, sersic_index=3.0
                ),
                mass=al.mp.dPIEMassSph(
                    centre=centre,
                    sigma=sigma,
                    r_core=0.0,
                    r_cut=10.0,
                    redshift_object=0.5,
                    redshift_source=1.0,
                ),
            )
            for centre, sigma in zip(SPECS["group"].extra_galaxies_centres, (212.0, 239.0))
        ]
        source = al.Galaxy(
            redshift=1.0,
            bulge=al.lp.SersicCore(
                centre=(0.0, 0.1),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                intensity=3.0,
                effective_radius=0.4,
                sersic_index=1.0,
            ),
        )
        return [lens_0, *extras, source], []

    raise KeyError(dataset_type)


#: The truth the cell model's narrow mass priors sit on: (centre, einstein_radius, axis_ratio,
#: angle) per main lens, shear (gamma_1, gamma_2), extra-galaxy dPIE sigma.
TRUTH = {
    "multi_galaxy": {
        "lenses": [((0.35, 0.25), 1.0, 0.85, 45.0), ((-0.35, -0.25), 0.8, 0.8, 120.0)],
        "lens_light_ell": [(0.8, 45.0), (0.7, 120.0)],
        "shear": (0.05, 0.05),
        "extra_sigma": (),
        "source": ((0.0, 0.03), 0.8, 60.0),
    },
    "group": {
        "lenses": [((0.0, 0.0), 4.0, 1.0, 0.0)],
        "lens_light_ell": [(1.0, 0.0)],
        "shear": (0.0, 0.0),
        "extra_sigma": (212.0, 239.0),
        "source": ((0.0, 0.1), 0.8, 60.0),
    },
}


def _light_priors_on_truth(basis, *, centre=None, axis_ratio=1.0, angle=0.0) -> None:
    """Re-centre an MGE basis's shared light priors on the truth (narrow Gaussians), one new
    prior per parameter shared by every Gaussian, as ``mge_model_from`` shares its own. The
    workspace's generic MGE priors put the iid draws' lens light ~0.1 off in ellipticity, a
    chi-squared of ~30 per pixel at hst depth: not the regime a sampler spends its time in."""
    import autofit as af
    import autolens as al

    ell = al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle)
    e0 = af.GaussianPrior(mean=ell[0], sigma=0.02)
    e1 = af.GaussianPrior(mean=ell[1], sigma=0.02)
    c0 = c1 = None
    if centre is not None:
        c0 = af.GaussianPrior(mean=centre[0], sigma=0.01)
        c1 = af.GaussianPrior(mean=centre[1], sigma=0.01)
    for gaussian in basis.profile_list:
        if hasattr(gaussian, "ell_comps"):
            gaussian.ell_comps.ell_comps_0 = e0
            gaussian.ell_comps.ell_comps_1 = e1
        if c0 is not None:
            gaussian.centre.centre_0 = c0
            gaussian.centre.centre_1 = c1


# --------------------------------------------------------------------------- simulation


def simulate(dataset_type: str, root: Path) -> Path:
    """Write ``dataset/<dataset_type>/<instrument>/`` (data, psf, noise map, tracer.json, centres).

    Deterministic (``noise_seed``); the submits call it under a lock before the first leg of an
    instrument, and the cell scripts call it when the dataset is missing.
    """
    import autolens as al

    spec = SPECS[dataset_type]
    path = dataset_path(dataset_type, root)
    path.mkdir(parents=True, exist_ok=True)
    grid = al.Grid2D.uniform(shape_native=spec.shape_native, pixel_scales=spec.pixel_scale)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=list(spec.simulate_sub_size_list),
        radial_list=list(spec.lp_radial_list),
        centre_list=spec.centres,
    )
    grid = grid.apply_over_sampling(over_sample_size=over_sample_size)
    psf = al.Convolver.from_gaussian(
        shape_native=spec.psf_shape, sigma=spec.psf_sigma, pixel_scales=grid.pixel_scales
    )
    simulator = al.SimulatorImaging(
        exposure_time=spec.exposure_time,
        psf=psf,
        background_sky_level=spec.background_sky_level,
        add_poisson_noise_to_data=True,
        noise_seed=spec.noise_seed,
    )
    galaxies, fields = truth_galaxies(dataset_type)
    tracer = al.Tracer(galaxies=galaxies, fields=fields) if fields else al.Tracer(galaxies=galaxies)
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    import autolens.plot as aplt

    aplt.fits_imaging(
        dataset=dataset,
        data_path=path / "data.fits",
        psf_path=path / "psf.fits",
        noise_map_path=path / "noise_map.fits",
        overwrite=True,
    )
    al.output_to_json(obj=tracer, file_path=path / "tracer.json")
    al.output_to_json(
        obj=al.Grid2DIrregular([list(c) for c in spec.main_lens_centres]),
        file_path=path / "main_lens_centres.json",
    )
    if spec.extra_galaxies_centres:
        al.output_to_json(
            obj=al.Grid2DIrregular([list(c) for c in spec.extra_galaxies_centres]),
            file_path=path / "extra_galaxies_centres.json",
        )
    (path / "simulation.json").write_text(
        json.dumps(
            {
                "dataset_type": dataset_type,
                "instrument": spec.instrument,
                "provenance": spec.provenance,
                "shape_native": list(spec.shape_native),
                "pixel_scale": spec.pixel_scale,
                "psf": {"shape": list(spec.psf_shape), "sigma": spec.psf_sigma},
                "exposure_time": spec.exposure_time,
                "background_sky_level": spec.background_sky_level,
                "noise_seed": spec.noise_seed,
                "over_sampling": {
                    "sub_size_list": list(spec.simulate_sub_size_list),
                    "radial_list": list(spec.lp_radial_list),
                    "centres": [list(c) for c in spec.centres],
                },
            },
            indent=2,
        )
    )
    return path


def simulate_if_missing(dataset_type: str, root: Path) -> Path:
    path = dataset_path(dataset_type, root)
    if not (path / "tracer.json").is_file() or not (path / "data.fits").is_file():
        print(f"  [simulate] {path} missing; simulating the {dataset_type} dataset")
        simulate(dataset_type, root)
    return path


def ensure_dataset(dataset_type: str, root: Path) -> Path:
    """:func:`simulate_if_missing` under a lock (array tasks of one dataset start together)."""
    import fcntl

    lock_dir = Path(root) / "output"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"p2d_dataset_{dataset_type}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return simulate_if_missing(dataset_type, root)


# --------------------------------------------------------------------------- the masked dataset


def load_masked(dataset_type: str, root: Path):
    """The simulated dataset, masked and (multi_galaxy) noise-scaled, light-profile over-sampled
    on every galaxy centre (workspace ``modeling.py``). Pixelization over-sampling is added by
    :func:`standard_pixelized_data` for the mesh variants."""
    import autolens as al

    spec = SPECS[dataset_type]
    path = ensure_dataset(dataset_type, root)
    dataset = al.Imaging.from_fits(
        data_path=path / "data.fits",
        psf_path=path / "psf.fits",
        noise_map_path=path / "noise_map.fits",
        pixel_scales=spec.pixel_scale,
    )
    if spec.noise_scaled is not None:
        centre, radius = spec.noise_scaled
        scaled = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            centre=centre,
            radius=radius,
        )
        # apply_noise_scaling scales the pixels where the mask is False, i.e. inside this circle.
        # (The workspace simulator writes this mask with invert=True, which under the current
        # library scales the complement instead: flagged on the campaign page, not mirrored.)
        dataset = dataset.apply_noise_scaling(mask=scaled)
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=spec.mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(over_sample_size_lp=lp_over_sample_size(spec, dataset))
    return dataset, path


def lp_over_sample_size(spec: DatasetSpec, dataset):
    import autolens as al

    return al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=list(spec.lp_sub_size_list),
        radial_list=list(spec.lp_radial_list),
        centre_list=spec.centres,
    )


def standard_preset(dataset_type: str, variant: str):
    """The hst production preset of ``variant`` at the standard size (decision D5), renamed and
    with this dataset type's light-profile over-sampling (bins on every galaxy centre)."""
    import _production_config as pc

    spec = SPECS[dataset_type]
    family = pc.STANDARD_FAMILY[variant]
    base = pc.standard_size_preset(family, pc.STANDARD_SIZE_REFERENCE_INSTRUMENT)
    note = (
        f"likelihood-lifecycle 2d (#473, decision D5): the {base.name} preset (hst, standard size) "
        f"on the {dataset_type} dataset at {spec.instrument}; light-profile over-sampling "
        f"{list(spec.lp_sub_size_list)} at {list(spec.lp_radial_list)} on every galaxy centre "
        "(workspace modeling.py); lens light = the workspace MGE (not the preset's); no positions "
        "penalty"
    )
    return replace(
        base,
        name=f"{base.name}_{dataset_type}",
        provenance=f"{base.provenance} — {note}",
        size_override=f"{base.size_override}; {note}" if base.size_override else note,
        lp_sub_size_list=tuple(spec.lp_sub_size_list),
        lp_radial_list=tuple(spec.lp_radial_list),
        positions_factor=None,
        positions_minimum_threshold=None,
    )


def standard_pixelized_data(dataset_type: str, dataset, path: Path, preset, sparse=None):
    """``(dataset, mesh adapt image, record)``: ``_production_config.production_standard_data``
    with the light-profile bins on every galaxy centre. The source adapt image is the subtracted
    S/N (every lens-plane galaxy of the truth tracer removed), the pixelization over-sampling is
    cut on the uncapped map and the mesh / regularization image capped at the preset's S/N."""
    import autolens as al
    import numpy as np

    import _production_config as pc

    spec = SPECS[dataset_type]
    lp = lp_over_sample_size(spec, dataset)
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=lp, over_sample_size_pixelization=preset.pix_low
    )
    snr, minimum_percent = pc.production_source_snr(dataset, path)
    above = np.asarray(snr) > preset.pix_snr_cut
    pix = al.Array2D(values=np.where(above, preset.pix_high, preset.pix_low), mask=dataset.mask)
    dataset = dataset.apply_over_sampling(over_sample_size_lp=lp, over_sample_size_pixelization=pix)
    dataset = apply_sparse(dataset, sparse)
    mesh_adapt = snr
    if preset.adapt_snr_cap is not None:
        mesh_adapt = al.Array2D(
            values=np.minimum(np.asarray(snr), preset.adapt_snr_cap), mask=dataset.mask
        )
    record = {
        "over_sampling_preset": preset.name,
        "over_sample_size_lp_rule": {
            "sub_size_list": list(spec.lp_sub_size_list),
            "radial_list": list(spec.lp_radial_list),
            "centres": [list(c) for c in spec.centres],
        },
        "over_sample_size_pixelization_rule": {
            "source_snr_cut": preset.pix_snr_cut,
            "sub_size_above": preset.pix_high,
            "sub_size_below": preset.pix_low,
            "threshold_map": "uncapped subtracted S/N (production)",
            "pixels_above": int(above.sum()),
            "pixels_below": int((~above).sum()),
        },
        "adapt_image": "subtracted S/N (truth tracer, every lens-plane galaxy removed), floored",
        "lens_light_source": "truth",
        "adapt_minimum_percent": minimum_percent,
        "mesh_adapt_snr_cap": preset.adapt_snr_cap,
    }
    return dataset, mesh_adapt, record


def apply_sparse(dataset, sparse):
    """``sparse``: ``"cpu"`` the numba CPU operator, an int the JAX operator at that batch size."""
    if sparse == "cpu":
        return dataset.apply_sparse_operator_cpu()
    if sparse:
        return dataset.apply_sparse_operator(batch_size=int(sparse))
    return dataset


# --------------------------------------------------------------------------- the cell model


def _dpie_constants(mass) -> None:
    """Fix the dPIE's non-parameters as the workspace does (``H0`` / ``Om0`` only where the
    pinned library has them)."""
    import autolens as al

    mass.r_core = 0.0
    mass.r_cut = 10.0
    mass.redshift_object = 0.5
    mass.redshift_source = 1.0
    params = inspect.signature(al.mp.dPIEMassSph.__init__).parameters
    if "H0" in params:
        mass.H0 = 67.66
    if "Om0" in params:
        mass.Om0 = 0.30966


def build_model(dataset_type: str, variant: str, *, source_pixelization=None):
    """The workspace modeling model (MGE form) with this variant's source.

    ``source_pixelization``: the ``al.Pixelization`` (or its ``af.Model``) of a mesh variant;
    None for ``mge`` (an MGE source of 20 Gaussians). Prior creation order is fixed (main lenses,
    shear, extra galaxies, source) so the seeded iid draws land on the same parameters.
    """
    import autofit as af
    import autolens as al

    spec = SPECS[dataset_type]
    truth = TRUTH[dataset_type]
    sigma_min = spec.pixel_scale / 10.0

    lenses = {}
    for i, (centre, einstein_radius, axis_ratio, angle) in enumerate(truth["lenses"]):
        bulge = al.model_util.mge_model_from(
            mask_radius=spec.mask_radius,
            total_gaussians=20,
            centre_prior_is_uniform=True,
            centre=tuple(centre),
            sigma_min=sigma_min,
        )
        light_q, light_angle = truth["lens_light_ell"][i]
        _light_priors_on_truth(bulge, centre=centre, axis_ratio=light_q, angle=light_angle)
        mass = af.Model(al.mp.Isothermal)
        if dataset_type == "multi_galaxy":
            mass.centre = tuple(centre)  # fixed to the light centre (multi_galaxy/modeling.py)
        else:
            mass.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=0.005)
            mass.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=0.005)
        ell = al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle)
        mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=ell[0], sigma=0.01)
        mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=ell[1], sigma=0.01)
        mass.einstein_radius = af.GaussianPrior(mean=einstein_radius, sigma=0.05)
        lenses[f"lens_{i}"] = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=truth["shear"][0], sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=truth["shear"][1], sigma=0.005)
    field = af.Model(al.MassField, redshift=0.5, shear=shear)

    extras = []
    for centre, sigma in zip(spec.extra_galaxies_centres, truth["extra_sigma"]):
        bulge = al.model_util.mge_model_from(
            mask_radius=spec.mask_radius,
            total_gaussians=10,
            centre_fixed=tuple(centre),
            sigma_min=sigma_min,
        )
        _light_priors_on_truth(bulge)  # SersicSph truth: round light
        mass = af.Model(al.mp.dPIEMassSph)
        mass.centre = tuple(centre)
        mass.sigma = af.GaussianPrior(mean=sigma, sigma=10.0)
        _dpie_constants(mass)
        extras.append(af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass))

    if source_pixelization is None:
        source_bulge = al.model_util.mge_model_from(
            mask_radius=spec.mask_radius,
            total_gaussians=20,
            gaussian_per_basis=1,
            centre_prior_is_uniform=False,
        )
        source_centre, source_q, source_angle = truth["source"]
        _light_priors_on_truth(
            source_bulge, centre=source_centre, axis_ratio=source_q, angle=source_angle
        )
        source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    else:
        source = af.Model(al.Galaxy, redshift=1.0, pixelization=source_pixelization)

    kwargs = {"galaxies": af.Collection(**lenses, source=source), "fields": field}
    if extras:
        kwargs["extra_galaxies"] = af.Collection(extras)
    return af.Collection(**kwargs)


def iid_instances(model, n: int, *, seed: int = IID_SEED, low=IID_UNIT_LOW, high=IID_UNIT_HIGH):
    """``n`` seeded iid draws from the central 20 % of every prior, with their unit vectors."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        unit = [float(u) for u in rng.uniform(low, high, size=model.prior_count)]
        out.append((model.instance_from_unit_vector(unit_vector=unit), unit))
    return out


def standard_cell(
    dataset_type: str,
    variant: str,
    *,
    root: Path,
    sparse=None,
    memo: str = "off",
    n_instances: int = 1,
    use_mixed_precision: bool = False,
):
    """Everything one setup of a galaxy/group likelihood needs, built the one shared way.

    ``sparse``: None (dense mapping), ``"cpu"`` (numba operator) or an int (JAX operator batch
    size). Returns a dict: ``spec``, ``dataset``, ``dataset_path``, ``preset`` (None for mge),
    ``adapt_image``, ``image_plane_mesh_grid``, ``n_source_pixels``,
    ``source_pixels_requested``, ``model``, ``instances`` / ``unit_vectors`` (the seeded iid
    stream; draw 0 is the cell's instance), ``settings``, ``memo_provenance``, ``record``.
    """
    import autolens as al

    import _production_config as pc

    if variant not in VARIANTS:
        raise KeyError(f"no {dataset_type} cell for variant {variant!r}")
    spec = SPECS[dataset_type]
    dataset, path = load_masked(dataset_type, root)
    record: dict = {
        "dataset_type": dataset_type,
        "instrument": spec.instrument,
        "dataset": spec.provenance,
        "mask_radius_arcsec": spec.mask_radius,
        "noise_scaled_region": (
            {"centre": list(spec.noise_scaled[0]), "radius": spec.noise_scaled[1]}
            if spec.noise_scaled
            else None
        ),
        "lens_light": "workspace MGE: 20 linear Gaussians per main lens"
        + (", 10 per extra galaxy (centre fixed)" if spec.extra_galaxies_centres else ""),
        "main_lenses": len(spec.main_lens_centres),
        "extra_galaxies": len(spec.extra_galaxies_centres),
        "ruling": "likelihood-lifecycle 2d draft (#473), decisions D1-D9 on the campaign page",
    }
    preset = grid = adapt_image = None
    if variant == "mge":
        dataset = apply_sparse(dataset, sparse)
        model = build_model(dataset_type, variant)
        n_source_pixels = requested = None
        # analysis_settings reads only the memo and precision here (no mesh preset for MGE).
        settings, memo_provenance = pc.analysis_settings(
            pc.preset_for("pixelization_numba", "hst"),
            memo,
            use_mixed_precision=use_mixed_precision,
        )
        record.update(
            source="MGE, 20 linear Gaussians (one basis)",
            sequence={
                "kind": "iid",
                "unit_low": IID_UNIT_LOW,
                "unit_high": IID_UNIT_HIGH,
                "seed": IID_SEED,
            },
        )
        seed, low, high = IID_SEED, IID_UNIT_LOW, IID_UNIT_HIGH
    else:
        preset = standard_preset(dataset_type, variant)
        dataset, adapt_image, data_record = standard_pixelized_data(
            dataset_type, dataset, path, preset, sparse=sparse
        )
        if preset.mesh_kind == "delaunay":
            grid = pc.production_image_mesh_grid(dataset, adapt_image, preset, spec.mask_radius)
            n_source_pixels = int(grid.shape[0])
            requested = pc.STANDARD_HILBERT_PIXELS
            mesh = pc.production_mesh(
                preset, n_vertices=n_source_pixels, mesh_class=al.mesh.Delaunay
            )
        else:
            n_source_pixels = preset.rect_pixels_yx**2
            requested = pc.STANDARD_RECT_PIXELS_YX**2
            mesh = pc.production_mesh(preset)
        import autofit as af

        reg = pc.regularization_model(preset)
        if isinstance(reg, af.Model) or isinstance(mesh, af.Model):
            pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=reg)
        else:
            pixelization = al.Pixelization(mesh=mesh, regularization=reg)
        model = build_model(dataset_type, variant, source_pixelization=pixelization)
        settings, memo_provenance = pc.analysis_settings(
            preset, memo, use_mixed_precision=use_mixed_precision
        )
        record.update(production_preset=preset.as_json(), production_data_record=data_record)
        seed, low, high = preset.seed, preset.iid_unit_low, preset.iid_unit_high
    draws = iid_instances(model, max(1, int(n_instances)), seed=seed, low=low, high=high)
    return {
        "spec": spec,
        "dataset": dataset,
        "dataset_path": path,
        "preset": preset,
        "adapt_image": adapt_image,
        "image_plane_mesh_grid": grid,
        "n_source_pixels": n_source_pixels,
        "source_pixels_requested": requested,
        "model": model,
        "instances": [d[0] for d in draws],
        "unit_vectors": [d[1] for d in draws],
        "settings": settings,
        "memo_provenance": memo_provenance,
        "record": record,
    }


def adapt_images_for(cell: dict, instance):
    """``al.AdaptImages`` for ``instance`` (None for mge: no adaptive component)."""
    import _production_config as pc

    if cell["adapt_image"] is None:
        return None
    return pc.adapt_images_for(instance, cell["image_plane_mesh_grid"], cell["adapt_image"])
