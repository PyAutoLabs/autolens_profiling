"""The seven-image Simple fixture from the reviewed autolens_visualization gallery."""

from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
GALLERY_REVISION = "017c8673aba5a579b2cba3318a3d1bae34b9b595"
INPUTS = ("point_datasets.csv", "scaling_galaxies.csv", "tracer.json")


def gaussian(value):
    import autofit as af

    return af.GaussianPrior(mean=float(value), sigma=max(abs(float(value)) * 0.01, 0.01))


def lens_model(lenses, *, free_host_mass=False):
    """Two free main galaxies, ten members with shared sigma_ref, and the NFW host."""
    import autofit as af
    import autolens as al

    galaxies = {}
    sigma_ref = gaussian(85.0)
    for i, galaxy in enumerate(lenses[:-1]):
        truth = galaxy.mass
        mass = af.Model(al.mp.dPIEMassSph)
        mass.redshift_object = truth.redshift_object
        mass.redshift_source = truth.redshift_source
        mass.H0, mass.Om0 = truth.H0, truth.Om0
        mass.r_core = truth.r_core
        if i < 2:
            name = f"main_{i}"
            mass.centre.centre_0 = gaussian(truth.centre[0])
            mass.centre.centre_1 = gaussian(truth.centre[1])
            mass.r_cut = gaussian(truth.r_cut)
            mass.sigma = gaussian(truth.sigma)
        else:
            name = f"scaling_{i - 2}"
            mass.centre = tuple(truth.centre)
            mass.r_cut = truth.r_cut
            mass.sigma = sigma_ref * (truth.sigma / 85.0)
        galaxies[name] = af.Model(al.Galaxy, redshift=galaxy.redshift, mass=mass)

    host = lenses[-1]
    truth = host.dark
    dark = af.Model(
        al.mp.NFWMCRLudlowSph,
        mass_at_200=af.LogUniformPrior(lower_limit=1e14, upper_limit=10**16.5)
        if free_host_mass
        else truth.mass_at_200,
        redshift_object=truth.redshift_object,
        redshift_source=truth.redshift_source,
    )
    dark.centre.centre_0 = gaussian(truth.centre[0])
    dark.centre.centre_1 = gaussian(truth.centre[1])
    galaxies["host_halo"] = af.Model(al.Galaxy, redshift=host.redshift, dark=dark)
    return af.Collection(galaxies=af.Collection(**galaxies))
