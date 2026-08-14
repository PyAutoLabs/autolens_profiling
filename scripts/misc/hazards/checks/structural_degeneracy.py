"""Measure a parameter direction that vanishes at the circular-profile edge."""

from __future__ import annotations

import numpy as np

from hazards._anchor import maybe_anchor_from_pattern
from hazards._likelihood import (
    DEFAULT_ELL_COMPS_SIGMA,
    ell_comps_radius_from_axis_ratio,
    imaging_pixelization_probe,
    isotropic_gaussian_disk_mass,
    nonfinite_gradient_site_persists,
    orientation_degeneracy_persists,
    orientation_spans,
)
from hazards._measure import Measurement, epsilon_neighbourhood_measurement
from hazards._record import Finding
from hazards.checks._base import HazardCheck, ScanContext


class StructuralDegeneracyCheck(HazardCheck):
    name = "structural_degeneracy"
    subject = "likelihood"

    def run(self, context: ScanContext) -> list[Finding]:
        probe = imaging_pixelization_probe(context)
        rows = [row for row in probe["structural"] if row.backend == "numpy"]
        spans = orientation_spans(rows)
        circular_span = spans.get(1.0, float("inf"))
        reference_span = spans.get(min(spans), 0.0)
        relative = circular_span / max(reference_span, 1.0e-14)
        axis_ratio_boundary = 0.99
        ell_comps_radius = ell_comps_radius_from_axis_ratio(axis_ratio_boundary)
        prior_mass = isotropic_gaussian_disk_mass(
            ell_comps_radius,
            sigma=DEFAULT_ELL_COMPS_SIGMA,
        )
        structural_finding = Finding(
            finding_id="likelihood.imaging-sersic.circular-orientation-degeneracy",
            title="Orientation vanishes at the circular profile boundary",
            summary=(
                "A complete imaging likelihood loses sensitivity to derived position angle "
                "at the Cartesian ell_comps origin. The q >= 0.99 neighbourhood contains "
                f"{prior_mass:.3e} of the default two-component Gaussian prior."
            ),
            hazard_class="structural_degeneracy",
            tier=2,
            subject="likelihood",
            subject_name="imaging_sersic",
            backends=("numpy",),
            measurements=(
                Measurement(
                    basis="error_curve",
                    value=relative,
                    unit="circular_over_elliptical_orientation_span",
                    details={
                        "axis_ratio_to_figure_of_merit_span": {
                            str(axis_ratio): span for axis_ratio, span in spans.items()
                        }
                    },
                ),
                epsilon_neighbourhood_measurement(
                    epsilon=ell_comps_radius,
                    mass=prior_mass,
                    domain={
                        "parameter": "ell_comps",
                        "dimensions": 2,
                        "type": "independent_gaussian",
                        "mean": 0.0,
                        "sigma": DEFAULT_ELL_COMPS_SIGMA,
                        "equivalent_axis_ratio_lower_bound": axis_ratio_boundary,
                    },
                    centre=(0.0, 0.0),
                ),
            ),
            anchors=(),
            code_exists=True,
            reachable_via=("FitImaging.Sersic.axis-ratio-angle-grid",),
            blocked_by=(),
            affects_science=False,
            backend_reachability={"numpy": {"full_likelihood": "reachable"}},
            reproducer={
                "axis_ratio_to_figure_of_merit_span": spans,
                "ell_comps_prior": {
                    "sigma": DEFAULT_ELL_COMPS_SIGMA,
                    "radius": ell_comps_radius,
                    "disk_mass": prior_mass,
                },
            },
        )
        sampled_parameters = ("ell_comps_0", "ell_comps_1")
        findings = (
            [structural_finding] if orientation_degeneracy_persists(sampled_parameters) else []
        )
        gradient_rows = probe.get("ell_comps_gradient", [])
        if gradient_rows:
            diagnostic = gradient_rows[0]
            gradients = tuple(tuple(values) for values in diagnostic["gradients"])
            origin_gradient = gradients[0]
            neighbourhood_gradients = gradients[1:]
            if nonfinite_gradient_site_persists(origin_gradient, neighbourhood_gradients):
                conversion_anchor = maybe_anchor_from_pattern(
                    context.workspace_root,
                    repo="PyAutoGalaxy",
                    path="autogalaxy/convert.py",
                    pattern="def axis_ratio_and_angle_from(",
                    after=49,
                    symbol="autogalaxy.convert.axis_ratio_and_angle_from",
                )
                gradient_norms = [
                    None if not np.all(np.isfinite(gradient)) else float(np.linalg.norm(gradient))
                    for gradient in gradients
                ]
                serialized_gradients = [
                    [float(value) if np.isfinite(value) else None for value in gradient]
                    for gradient in gradients
                ]
                gradient_epsilon = 1.0e-8
                gradient_neighbourhood_mass = isotropic_gaussian_disk_mass(
                    gradient_epsilon,
                    sigma=DEFAULT_ELL_COMPS_SIGMA,
                )
                findings.append(
                    Finding(
                        finding_id=(
                            "likelihood.imaging-sersic.ell-comps-origin-nonfinite-gradient"
                        ),
                        title="Circular ell_comps origin has a non-finite JAX gradient",
                        summary=(
                            "The q-angle counterfactual is not sampler-reachable because Sersic "
                            "fits ell_comps_0/1. In those actual coordinates the off-centre "
                            "complete likelihood is finite at the origin, but its exact JAX "
                            "gradient is non-finite while nearby gradients are finite."
                        ),
                        hazard_class="nonfinite_gradient",
                        tier=2,
                        subject="likelihood",
                        subject_name="imaging_sersic_ell_comps",
                        backends=("jax",),
                        measurements=(
                            Measurement(
                                basis="epsilon_neighbourhood",
                                value=gradient_neighbourhood_mass,
                                unit="fraction",
                                details={
                                    "epsilon": gradient_epsilon,
                                    "centre": [0.0, 0.0],
                                    "domain": {
                                        "parameter": "ell_comps",
                                        "dimensions": 2,
                                        "type": "independent_gaussian",
                                        "sigma": DEFAULT_ELL_COMPS_SIGMA,
                                        "note": "the exact point has zero continuous prior mass",
                                    },
                                },
                            ),
                            Measurement(
                                basis="error_curve",
                                value=None,
                                unit="gradient_norm",
                                details={
                                    "points": diagnostic["points"],
                                    "gradient_norm": gradient_norms,
                                    "origin_gradient": serialized_gradients[0],
                                },
                            ),
                        ),
                        anchors=(conversion_anchor,) if conversion_anchor is not None else (),
                        code_exists=True,
                        reachable_via=(
                            "FitImaging.Sersic.ell-comps-prior-mean",
                            "jax_autodiff",
                        ),
                        blocked_by=(),
                        affects_science=True,
                        backend_reachability={
                            "jax": {
                                "full_likelihood": "reachable",
                                "value": "finite",
                                "gradient": "nonfinite_at_exact_origin",
                            }
                        },
                        reproducer={
                            "parameter": "ell_comps_radius",
                            "radius": [
                                float(np.linalg.norm(point)) for point in diagnostic["points"]
                            ],
                            "gradient_norm": gradient_norms,
                            "points": diagnostic["points"],
                            "figure_of_merit": diagnostic["figure_of_merit"],
                            "gradient": serialized_gradients,
                            "gradient_neighbourhood_prior_mass": (gradient_neighbourhood_mass),
                            "recommendation": (
                                "Open a bounded PyAutoGalaxy task to make the Cartesian "
                                "ell_comps-to-geometry path differentiable at the origin "
                                "without assigning an arbitrary physical orientation."
                            ),
                            "resolved_structural_counterfactual": {
                                "finding_id": (
                                    "likelihood.imaging-sersic.circular-orientation-degeneracy"
                                ),
                                "sampled_parameters": list(sampled_parameters),
                                "axis_ratio_to_figure_of_merit_span": spans,
                                "q_greater_equal_0.99_prior_mass": prior_mass,
                                "reason": (
                                    "angle is derived from Cartesian ell_comps, not an "
                                    "independent sampled direction"
                                ),
                            },
                        },
                    )
                )
        return findings
