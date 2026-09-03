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

from collections.abc import Callable
from dataclasses import dataclass

import autolens as al
import numpy as np

# Axis ratio 0.8 at 45 degrees, shared by every elliptical fiducial so the
# elliptical / spherical pairs differ only in the geometry code path.
_ELL = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)
ELL_COMPS: tuple[float, float] = (float(_ELL[0]), float(_ELL[1]))

CENTRE: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ProfileSpec:
    """One profile under measurement: what to build, and how to describe it."""

    name: str
    family: str  # "total" | "dark" | "stellar" | "basis" — matches the cell filename
    cls_name: str  # attribute on ``al.mp``, or the described class for a factory spec
    params: dict
    factory: Callable[..., object] | None = None

    def build(self):
        """Instantiate the profile from its fiducial parameters.

        A spec with a ``factory`` builds through it instead of ``al.mp``: a ``Basis``
        is not an attribute of ``al.mp`` and its components are derived from the
        parameters rather than passed straight through.
        """
        if self.factory is not None:
            return self.factory(**self.params)
        return getattr(al.mp, self.cls_name)(**self.params)

    @property
    def json_params(self) -> dict:
        """JSON-serialisable copy of ``params`` (tuples become lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()}


def _spec(name: str, family: str, cls_name: str, **params) -> ProfileSpec:
    return ProfileSpec(name=name, family=family, cls_name=cls_name, params=params)


def mge_basis(
    *,
    total_gaussians: int,
    sigma_min: float,
    sigma_max: float,
    axis_ratio: float,
    angle: float,
    intensity: float,
    mass_to_light_ratio: float,
) -> al.lp_basis.Basis:
    """A ``Basis`` of light-and-mass Gaussians with log-spaced widths.

    The shape of an MGE lens light in the SLaM ``mass_light_dark`` stage
    (``autogalaxy/analysis/chaining_util.py``): every Gaussian's centre, ellipticity,
    intensity and sigma is fixed by the light stage, and the whole stack shares **one**
    ``mass_to_light_ratio``. The sigma spacing mirrors
    ``mge_model_from`` — log-spaced from ``sigma_min`` to the mask radius.
    """
    ell = al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle)

    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), total_gaussians)

    return al.lp_basis.Basis(
        profile_list=[
            al.lmp.Gaussian(
                centre=CENTRE,
                ell_comps=(float(ell[0]), float(ell[1])),
                intensity=intensity,
                sigma=float(sigma),
                mass_to_light_ratio=mass_to_light_ratio,
            )
            for sigma in sigmas
        ]
    )


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
    # --- basis (the MGE stack the fixed-geometry deflection memo is built for) ---
    "Basis_mge_30": ProfileSpec(
        name="Basis_mge_30",
        family="basis",
        cls_name="Basis",
        params={
            "total_gaussians": 30,
            "sigma_min": 0.01,
            "sigma_max": 3.5,
            "axis_ratio": 0.8,
            "angle": 45.0,
            "intensity": 1.0,
            "mass_to_light_ratio": 1.0,
        },
        factory=mge_basis,
    ),
}
