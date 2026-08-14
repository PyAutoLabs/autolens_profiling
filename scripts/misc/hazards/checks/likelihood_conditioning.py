"""Measure inversion floors and a scale-aware counterfactual in a real fit."""

from __future__ import annotations

import inspect
import math
import re

from hazards._anchor import maybe_anchor_from_pattern
from hazards._likelihood import (
    conditioning_policy_metrics,
    floor_fraction,
    imaging_pixelization_probe,
)
from hazards._measure import Measurement, reachability_measurement
from hazards._record import Finding
from hazards.checks._base import HazardCheck, ScanContext

_FLOAT_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_PACKAGED_DEFAULT_PATTERN = re.compile(
    rf"packaged\s+configuration\s+defaults?\s+to\s+`{{0,2}}({_FLOAT_PATTERN})",
    re.IGNORECASE,
)
_LEGACY_HELPER_DEFAULT_PATTERN = re.compile(
    rf"small\s+numerical\s+value\s+of\s+`{{0,2}}({_FLOAT_PATTERN})",
    re.IGNORECASE,
)


def curvature_floor_documentation_defaults(
    helper_documentation: str | None,
    settings_documentation: str | None,
) -> dict[str, float | None]:
    """Extract the packaged curvature-floor defaults claimed by both docs."""

    helper_text = helper_documentation or ""
    settings_text = settings_documentation or ""
    helper_match = _PACKAGED_DEFAULT_PATTERN.search(helper_text)
    if helper_match is None:
        helper_match = _LEGACY_HELPER_DEFAULT_PATTERN.search(helper_text)
    settings_match = _PACKAGED_DEFAULT_PATTERN.search(settings_text)
    return {
        "helper": float(helper_match.group(1)) if helper_match is not None else None,
        "settings": float(settings_match.group(1)) if settings_match is not None else None,
    }


def curvature_floor_documentation_drift_persists(
    documented_defaults: dict[str, float | None],
    *,
    configured_floor: float,
) -> bool:
    """Return whether either public doc is missing or disagrees with configuration."""

    if not math.isfinite(configured_floor) or configured_floor <= 0.0:
        return True
    return any(
        value is None
        or not math.isfinite(value)
        or not math.isclose(value, configured_floor, rel_tol=1.0e-12, abs_tol=0.0)
        for value in documented_defaults.values()
    )


class LikelihoodConditioningCheck(HazardCheck):
    name = "likelihood_conditioning"
    subject = "likelihood"

    def run(self, context: ScanContext) -> list[Finding]:
        from autoarray.inversion.inversion import inversion_util
        from autoarray.settings import Settings

        probe = imaging_pixelization_probe(context)
        rows = sorted(
            (row for row in probe["inversion"] if row.backend == "numpy" and row.parameter == 0.9),
            key=lambda row: row.noise_scale,
        )
        configured_floor = float(Settings().no_regularization_add_to_curvature_diag_value)
        regularization_jitter = 1.0e-8
        policy_metrics = conditioning_policy_metrics(probe["conditioning"])
        absolute = policy_metrics["absolute"]
        scale_aware = policy_metrics["scale_aware"]
        no_floor = policy_metrics["none"]
        curvature_ratios = absolute["floor_fraction"]
        regularization_ratios = [
            floor_fraction(regularization_jitter, row.regularization_diagonal) for row in rows
        ]
        scale_aware_span = max(scale_aware["floor_fraction"]) - min(scale_aware["floor_fraction"])
        scale_aware_output_error = max(
            scale_aware["figure_of_merit_relative_error"]
            + scale_aware["reconstruction_relative_error"]
        )

        curvature_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/inversion/inversion/inversion_util.py",
            pattern="def curvature_matrix_with_added_to_diag_from(",
            after=18,
            symbol="autoarray.inversion.inversion.inversion_util.curvature_matrix_with_added_to_diag_from",
        )
        config_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/config/general.yaml",
            pattern="no_regularization_add_to_curvature_diag_value :",
            config_key="general.inversion.no_regularization_add_to_curvature_diag_value",
        )
        constant_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/inversion/regularization/constant.py",
            pattern="diag_vals = 1e-8 +",
            after=3,
            symbol="autoarray.inversion.regularization.constant.constant_regularization_matrix_from",
        )
        gaussian_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/inversion/regularization/gaussian_kernel.py",
            pattern="h_jitter = 1e-8 * xp.abs(diag_mean)",
            after=3,
            symbol="autoarray.inversion.regularization.gaussian_kernel.GaussianKernel.regularization_matrix_from",
        )
        reachability = reachability_measurement(
            reachable_via=["FitImaging.linear-light-plus-RectangularUniform.Constant"]
        )
        findings = [
            Finding(
                finding_id="likelihood.imaging-pixelization.absolute-conditioning-floors",
                title="Absolute inversion floors move with dataset scale",
                summary=(
                    "Using only the diagonal entries the policy actually touches, the "
                    f"absolute floor reaches {max(curvature_ratios):.3e} of their scale. "
                    "A reference-calibrated scale-aware counterfactual holds its fraction "
                    f"fixed with maximum relative output error {scale_aware_output_error:.3e}."
                ),
                hazard_class="conditioning_floor",
                tier=2,
                subject="likelihood",
                subject_name="imaging_pixelization",
                backends=("numpy",),
                measurements=(
                    Measurement(
                        basis="error_curve",
                        value=max(curvature_ratios),
                        unit="curvature_floor_over_diagonal_scale",
                        details={
                            "noise_scale": [row.noise_scale for row in rows],
                            "fraction": curvature_ratios,
                            "absolute_floor": configured_floor,
                            "conditioned_indices": rows[0].metadata["conditioned_indices"],
                            "denominator": (
                                "median absolute curvature diagonal at no_regularization_index_list"
                            ),
                        },
                    ),
                    Measurement(
                        basis="error_curve",
                        value=max(regularization_ratios),
                        unit="regularization_jitter_over_diagonal_scale",
                        details={
                            "noise_scale": [row.noise_scale for row in rows],
                            "fraction": regularization_ratios,
                            "absolute_jitter": regularization_jitter,
                            "scale_free_counterexample": "GaussianKernel trace-scaled h_jitter",
                        },
                    ),
                    Measurement(
                        basis="error_curve",
                        value=scale_aware_span,
                        unit="scale_aware_curvature_floor_fraction_span",
                        details={
                            "noise_scale": scale_aware["noise_scale"],
                            "fraction": scale_aware["floor_fraction"],
                            "floor_value": scale_aware["floor_value"],
                            "calibration": "matches absolute default at noise_scale=1.0",
                        },
                    ),
                    Measurement(
                        basis="error_curve",
                        value=scale_aware_output_error,
                        unit="max_relative_output_error_vs_absolute_policy",
                        details={
                            "scale_aware": scale_aware,
                            "zero_floor_control": no_floor,
                        },
                    ),
                    reachability,
                ),
                anchors=tuple(
                    anchor
                    for anchor in (
                        curvature_anchor,
                        config_anchor,
                        constant_anchor,
                        gaussian_anchor,
                    )
                    if anchor is not None
                ),
                code_exists=True,
                reachable_via=("FitImaging.linear-light-plus-RectangularUniform.Constant",),
                blocked_by=(),
                affects_science=None,
                backend_reachability={"numpy": {"full_likelihood": "reachable"}},
                reproducer={
                    "noise_scale": [row.noise_scale for row in rows],
                    "curvature_floor_fraction": curvature_ratios,
                    "scale_aware_curvature_floor_fraction": scale_aware["floor_fraction"],
                    "regularization_jitter_fraction": regularization_ratios,
                    "conditioning_policies": policy_metrics,
                    "zero_floor_solvable": True,
                    "phase_2_denominator_correction": (
                        "The earlier 11.5% headline used the median of the full "
                        "matrix. The floor only touches no_regularization_index_list; "
                        "this record measures those affected entries."
                    ),
                    "recommendation": (
                        "Do not change the PyAutoArray default from this fixture. "
                        "Scale dependence is real, but the maximum affected-entry "
                        "fraction is small; require representative workspace evidence "
                        "before opening a source-numerics task."
                    ),
                },
            )
        ]
        documented_defaults = curvature_floor_documentation_defaults(
            inspect.getdoc(inversion_util.curvature_matrix_with_added_to_diag_from),
            inspect.getdoc(Settings.__init__),
        )
        if curvature_floor_documentation_drift_persists(
            documented_defaults,
            configured_floor=configured_floor,
        ):
            documented_reference = next(
                (
                    value
                    for value in documented_defaults.values()
                    if value is not None and math.isfinite(value) and value > 0.0
                ),
                configured_floor,
            )
            findings.append(
                Finding(
                    finding_id="likelihood.imaging-pixelization.curvature-floor-doc-config-drift",
                    title="Curvature-floor documentation trails the live default",
                    summary=(
                        "The helper and Settings documentation do not both name the live "
                        f"{configured_floor:.1e} packaged curvature-floor default."
                    ),
                    hazard_class="documentation_drift",
                    tier=2,
                    subject="likelihood",
                    subject_name="imaging_pixelization",
                    backends=tuple(context.backends),
                    measurements=(
                        Measurement(
                            basis="error_curve",
                            value=configured_floor / documented_reference,
                            unit="configured_over_documented",
                            details={
                                "documented": documented_defaults,
                                "configured": configured_floor,
                            },
                        ),
                    ),
                    anchors=tuple(
                        anchor for anchor in (curvature_anchor, config_anchor) if anchor is not None
                    ),
                    code_exists=True,
                    reachable_via=("FitImaging.linear-light-inversion",),
                    blocked_by=(),
                    affects_science=None,
                    reproducer={
                        "documented": documented_defaults,
                        "configured": configured_floor,
                    },
                )
            )
        return findings
