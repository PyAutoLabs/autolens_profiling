"""Region-seam derivative jumps and the spherical q clamp in the MGE deflection path."""

from __future__ import annotations

import numpy as np

from hazards._anchor import maybe_anchor_from_pattern
from hazards._measure import error_curve_measurement, reachability_measurement
from hazards._record import Finding
from hazards.checks._base import HazardCheck, ScanContext
from hazards.mge_faddeeva import (
    CLAMP_FINDING_ID,
    INNER_SLOPE,
    KAPPA_S,
    SCALE_RADIUS,
    SEAM_FINDING_ID,
    seam_derivative_discontinuity,
    seam_error_curve_arrays,
)

# A region-selecting routine's derivative steps across a boundary by far more than
# the derivative genuinely moves there; a single smooth expression steps by exactly
# the genuine variation. `seam_derivative_discontinuity` returns that ratio, so the
# gate is dimensionless and needs no tuning to a routine's accuracy: measured 2.28e4
# for the former `_wofz_rational` and 1.000 for the Weideman series that replaced it
# (PyAutoGalaxy#600 phase B), and any gate between 10 and 1e3 separates them.
SEAM_EXCESS_FACTOR_GATE = 100.0

# The signature of evaluating a circular profile as an ellipse: a deflection that is
# not purely radial. The exact spherical branch is radial by construction, so this is
# exactly zero there; the q = 0.9999 clamp left up to 1.4e-4 arcsec.
CLAMP_CROSS_AXIS_GATE_ARCSEC = 1.0e-12

SAMPLE_COORDINATES = (
    (0.0, 1.0),
    (0.7071067811865476, 0.7071067811865476),
    (1.0, 0.0),
    (0.0, -1.0),
    (0.0, 0.25),
    (0.25, 0.0),
    (1.5, 1.5),
    (-2.0, 0.5),
)


class MGEFaddeevaCheck(HazardCheck):
    """Detect the JAX Faddeeva seams and the ``q = 0.9999`` spherical clamp."""

    name = "mge_faddeeva"
    subject = "component"

    def run(self, context: ScanContext) -> list[Finding]:
        import autoarray as aa
        import autolens as al
        import jax
        import jax.numpy as jnp
        from autogalaxy.profiles.mass.abstract.mge import _wofz_weideman

        jax.config.update("jax_enable_x64", True)

        seam_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoGalaxy",
            path="autogalaxy/profiles/mass/abstract/mge.py",
            pattern="denominator = _WOFZ_WEIDEMAN_L - 1j * z",
            after=6,
            symbol="autogalaxy.profiles.mass.abstract.mge._wofz_weideman",
        )
        mge_clamp_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoGalaxy",
            path="autogalaxy/profiles/mass/abstract/mge.py",
            pattern="return xp.where(axis_ratio < 0.9999, axis_ratio, 0.9999)",
            before=2,
            symbol="autogalaxy.profiles.mass.abstract.mge.MGEDecomposer.axis_ratio",
        )
        gaussian_clamp_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoGalaxy",
            path="autogalaxy/profiles/mass/stellar/gaussian.py",
            pattern="return xp.where(axis_ratio < 0.9999, axis_ratio, 0.9999)",
            before=2,
            symbol="autogalaxy.profiles.mass.stellar.gaussian.Gaussian.axis_ratio",
        )
        spherical_branch_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoGalaxy",
            path="autogalaxy/profiles/mass/abstract/mge.py",
            pattern="if _is_circular(self.ell_comps):",
            after=1,
            symbol="autogalaxy.profiles.mass.abstract.mge.MGEDecomposer.deflections_2d_via_mge_from",
        )

        findings: list[Finding] = []

        offsets, exact_derivative, autodiff_derivative = seam_error_curve_arrays(_wofz_weideman)
        seam_measurement = error_curve_measurement(
            offsets,
            exact_derivative,
            autodiff_derivative,
            parameter_name="relative offset above the r2=2.5 region seam",
        )
        seam_step = seam_derivative_discontinuity(_wofz_weideman)
        if seam_step["excess_factor"] > SEAM_EXCESS_FACTOR_GATE:
            findings.append(
                Finding(
                    finding_id=SEAM_FINDING_ID,
                    title="The JAX Faddeeva routine's region seams make its derivative discontinuous",
                    summary=(
                        "The JAX Faddeeva routine selects one of several rational branches with "
                        "`xp.where` on |z|^2 and Im(z)^2. The value is continuous to ~1e-6 across a "
                        "seam but the derivative is not, so `jax.grad` of any MGE deflection carries "
                        "a kink wherever a grid point crosses a boundary. Straddling the r2=2.5 seam "
                        f"at a relative offset of 1e-9 the derivative jumps "
                        f"{seam_step['excess_factor']:.3g}x further than it genuinely moves there, "
                        f"and the autodiff derivative error reaches {seam_measurement.value:.3g} "
                        "relative against 1.3e-14 for the SciPy path the numpy backend uses."
                    ),
                    hazard_class="nonsmooth_objective",
                    tier=1,
                    subject="component",
                    subject_name="mge",
                    backends=("jax",),
                    measurements=(
                        seam_measurement,
                        reachability_measurement(
                            reachable_via=[
                                "jax_mge_deflections",
                                "jax_gaussian_deflections",
                            ]
                        ),
                    ),
                    anchors=tuple(anchor for anchor in (seam_anchor,) if anchor is not None),
                    code_exists=True,
                    reachable_via=("jax_mge_deflections", "jax_gaussian_deflections"),
                    blocked_by=(),
                    affects_science=True,
                    backend_reachability={
                        "numpy": {"implementation": "scipy.special.wofz", "seams": 0},
                        "jax": {
                            "implementation": "_wofz_weideman",
                            "region_boundaries": [
                                "r2=2.5",
                                "r2=30",
                                "r2=62",
                                "y2=0.072",
                                "y2=1e-13",
                            ],
                        },
                    },
                    reproducer={
                        "seam": "r2=2.5",
                        "angle": 0.0,
                        "offsets": offsets.tolist(),
                        "relative_error": seam_measurement.details["relative_error"],
                        "derivative_discontinuity": seam_step,
                        "study": "scripts/misc/hazards/mge_faddeeva.py",
                    },
                )
            )

        coordinates = np.asarray(SAMPLE_COORDINATES, dtype=float)
        profile = al.mp.gNFWSph(
            centre=(0.0, 0.0),
            kappa_s=KAPPA_S,
            inner_slope=INNER_SLOPE,
            scale_radius=SCALE_RADIUS,
        )
        exact = np.asarray(
            profile.deflections_yx_2d_from(
                grid=aa.Grid2DIrregular(values=coordinates), xp=np
            ).array,
            dtype=float,
        )
        clamped = np.asarray(
            profile.deflections_yx_2d_from(
                grid=aa.Grid2DIrregular(values=jnp.asarray(coordinates)), xp=jnp
            ).array,
            dtype=float,
        )
        radii = np.hypot(coordinates[:, 0], coordinates[:, 1])
        clamp_measurement = error_curve_measurement(
            radii,
            exact,
            clamped,
            parameter_name="radius_arcsec",
        )
        cross_axis = (
            clamped[:, 0] * coordinates[:, 1] - clamped[:, 1] * coordinates[:, 0]
        ) / np.maximum(radii, 1.0e-300)
        max_cross_axis = float(np.max(np.abs(cross_axis)))
        if max_cross_axis > CLAMP_CROSS_AXIS_GATE_ARCSEC:
            findings.append(
                Finding(
                    finding_id=CLAMP_FINDING_ID,
                    title="Spherical MGE profiles are evaluated as ellipses at the q = 0.9999 clamp on JAX",
                    summary=(
                        "The elliptical Faddeeva form is singular at q = 1, so the MGE machinery "
                        "clamps the axis ratio to 0.9999. The numpy path escapes through an exact "
                        "radial branch; the JAX path cannot, and every *Sph MGE-routed profile is "
                        f"biased by up to {clamp_measurement.value:.3g} relative with a spurious "
                        f"cross-axis deflection of {max_cross_axis:.3g} arcsec -- a circular "
                        "profile's deflection is purely radial, so any cross-axis term at all is "
                        "the ellipse the clamp forced it into."
                    ),
                    hazard_class="backend_divergence",
                    tier=1,
                    subject="component",
                    subject_name="mge",
                    backends=("numpy", "jax"),
                    measurements=(
                        clamp_measurement,
                        reachability_measurement(
                            reachable_via=["jax_mge_deflections", "jax_gaussian_deflections"],
                            blocked_paths={
                                "numpy_mge_deflections": "exact spherical branch (_spherical_mge_deflections_from)"
                            },
                        ),
                    ),
                    anchors=tuple(
                        anchor
                        for anchor in (
                            mge_clamp_anchor,
                            gaussian_clamp_anchor,
                            spherical_branch_anchor,
                        )
                        if anchor is not None
                    ),
                    code_exists=True,
                    reachable_via=("jax_mge_deflections", "jax_gaussian_deflections"),
                    blocked_by=(),
                    affects_science=True,
                    backend_reachability={
                        "numpy": {"implementation": "exact spherical branch", "axis_ratio": 1.0},
                        "jax": {"implementation": "elliptical Faddeeva", "axis_ratio": 0.9999},
                    },
                    reproducer={
                        "max_cross_axis_deflection_arcsec": max_cross_axis,
                        "profile": "gNFWSph",
                        "kappa_s": KAPPA_S,
                        "inner_slope": INNER_SLOPE,
                        "scale_radius": SCALE_RADIUS,
                        "coordinates": coordinates.tolist(),
                        "relative_error": clamp_measurement.details["relative_error"],
                        "cross_axis_deflection": cross_axis.tolist(),
                        "study": "scripts/misc/hazards/mge_faddeeva.py",
                    },
                )
            )
        return findings
