"""Registry of the mass profiles measured by the deflection-angle cells.

One place holds the *identity* of every profile under measurement — its class,
its fiducial parameters and the family (cell) it belongs to — so the three thin
cells (``total.py`` / ``dark.py`` / ``stellar.py``) never disagree about what
"Isothermal" means, and so the recorded JSON can embed the exact parameters that
produced a timing.

Fiducials are deliberately *physical but unremarkable*: an einstein radius of
1.6" and an axis ratio of 0.8 at 45 degrees for the total-mass profiles, a
kappa_s of 0.2 with a 10" scale radius for the dark profiles, and unit intensity
with sigma 1.0" for the stellar Gaussian. Nothing here is tuned to a dataset —
the deflection cost of a numpy mass profile depends on the *grid size* and the
profile's own math, not on the data.

Note on the spherical Gaussian: ``autogalaxy.mp`` has **no** ``GaussianSph``.
The spherical case is the elliptical ``Gaussian`` with ``ell_comps=(0.0, 0.0)``,
registered here as ``Gaussian_sph_case`` so the two rows stay distinguishable in
the dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass

import autolens as al

# Axis ratio 0.8 at 45 degrees, shared by every elliptical fiducial so the
# elliptical / spherical pairs differ only in the geometry code path.
_ELL = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)
ELL_COMPS: tuple[float, float] = (float(_ELL[0]), float(_ELL[1]))

CENTRE: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ProfileSpec:
    """One profile under measurement: what to build, and how to describe it."""

    name: str
    family: str  # "total" | "dark" | "stellar" — matches the cell filename
    cls_name: str  # attribute on ``al.mp``
    params: dict

    def build(self):
        """Instantiate the profile from its fiducial parameters."""
        return getattr(al.mp, self.cls_name)(**self.params)

    @property
    def json_params(self) -> dict:
        """JSON-serialisable copy of ``params`` (tuples become lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()}


def _spec(name: str, family: str, cls_name: str, **params) -> ProfileSpec:
    return ProfileSpec(name=name, family=family, cls_name=cls_name, params=params)


PROFILES: dict[str, ProfileSpec] = {
    # --- total (mass follows light + dark, the lens-modelling workhorses) ---
    "Isothermal": _spec(
        "Isothermal",
        "total",
        "Isothermal",
        centre=CENTRE,
        ell_comps=ELL_COMPS,
        einstein_radius=1.6,
    ),
    "IsothermalSph": _spec(
        "IsothermalSph",
        "total",
        "IsothermalSph",
        centre=CENTRE,
        einstein_radius=1.6,
    ),
    "PowerLaw": _spec(
        "PowerLaw",
        "total",
        "PowerLaw",
        centre=CENTRE,
        ell_comps=ELL_COMPS,
        einstein_radius=1.6,
        slope=2.2,
    ),
    "PowerLawSph": _spec(
        "PowerLawSph",
        "total",
        "PowerLawSph",
        centre=CENTRE,
        einstein_radius=1.6,
        slope=2.2,
    ),
    # --- dark (NFW family) ---
    "NFW": _spec(
        "NFW",
        "dark",
        "NFW",
        centre=CENTRE,
        ell_comps=ELL_COMPS,
        kappa_s=0.2,
        scale_radius=10.0,
    ),
    "NFWSph": _spec(
        "NFWSph",
        "dark",
        "NFWSph",
        centre=CENTRE,
        kappa_s=0.2,
        scale_radius=10.0,
    ),
    "gNFW": _spec(
        "gNFW",
        "dark",
        "gNFW",
        centre=CENTRE,
        ell_comps=ELL_COMPS,
        kappa_s=0.2,
        inner_slope=1.5,
        scale_radius=10.0,
    ),
    "gNFWSph": _spec(
        "gNFWSph",
        "dark",
        "gNFWSph",
        centre=CENTRE,
        kappa_s=0.2,
        inner_slope=1.5,
        scale_radius=10.0,
    ),
    # --- stellar (the Gaussian basis component; MGE bases are stacks of these) ---
    "Gaussian": _spec(
        "Gaussian",
        "stellar",
        "Gaussian",
        centre=CENTRE,
        ell_comps=ELL_COMPS,
        intensity=1.0,
        sigma=1.0,
    ),
    "Gaussian_sph_case": _spec(
        "Gaussian_sph_case",
        "stellar",
        "Gaussian",
        centre=CENTRE,
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=1.0,
    ),
}
