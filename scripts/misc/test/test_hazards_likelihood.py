"""Pure likelihood-tier analysis tests; no PyAuto/JAX imports required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in sys.path:
    sys.path.insert(0, str(_misc))

from hazards._likelihood import (  # noqa: E402
    LikelihoodProbeRow,
    backend_divergence_persists,
    backend_error_curves,
    calibrated_scale_aware_floors,
    conditioning_policy_metrics,
    epsilon_neighbourhood_mass,
    floor_fraction,
    nnls_optimality_metrics,
    orientation_spans,
    relative_l2_error,
    stable_ellipse_parameters_from_border,
    support_transition_locations,
)


def _row(parameter, reconstruction, *, backend="numpy", figure_of_merit=-10.0, **metadata):
    return LikelihoodProbeRow(
        parameter=parameter,
        parameter_name="einstein_radius",
        backend=backend,
        figure_of_merit=figure_of_merit,
        reconstruction=tuple(reconstruction),
        metadata=metadata,
    )


def test_active_support_transitions_are_midpoints_not_exact_prior_mass():
    rows = [
        _row(0.0, (1.0, 0.0)),
        _row(1.0, (0.8, 0.2)),
        _row(2.0, (0.0, 1.0)),
    ]
    assert support_transition_locations(rows) == [0.5, 1.5]
    assert epsilon_neighbourhood_mass(
        [0.5, 1.5], epsilon=0.25, lower=0.0, upper=2.0
    ) == pytest.approx(0.5)


def test_epsilon_neighbourhood_mass_merges_overlap_and_clips_edges():
    assert epsilon_neighbourhood_mass(
        [0.0, 0.1, 1.0], epsilon=0.2, lower=0.0, upper=1.0
    ) == pytest.approx(0.5)


def test_backend_curves_align_parameters_and_report_both_outputs():
    rows = [
        _row(1.0, (1.0, 2.0), figure_of_merit=-10.0),
        _row(1.0, (1.1, 1.9), backend="jax", figure_of_merit=-9.0),
    ]
    curve = backend_error_curves(rows)["jax"]
    assert curve["parameter"] == [1.0]
    assert curve["figure_of_merit"] == pytest.approx([0.1])
    assert curve["reconstruction"][0] > 0.0


def test_backend_divergence_persistence_uses_relative_parity_tolerance():
    curves = {
        "jax": {
            "parameter": [1.0],
            "figure_of_merit": [1.0e-12],
            "reconstruction": [9.0e-9],
        }
    }
    assert backend_divergence_persists(curves) is False
    curves["jax"]["reconstruction"] = [1.1e-8]
    assert backend_divergence_persists(curves) is True
    curves["jax"]["reconstruction"] = [float("nan")]
    assert backend_divergence_persists(curves) is True


def test_backend_divergence_persistence_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError, match="relative tolerance must be positive"):
        backend_divergence_persists({}, relative_tolerance=0.0)


def test_nnls_metrics_grade_primal_dual_and_complementarity_conditions():
    matrix = np.diag([2.0, 1.0])
    vector = np.array([2.0, -1.0])
    optimum = nnls_optimality_metrics(matrix, vector, np.array([1.0, 0.0]))
    assert optimum == pytest.approx(
        {
            "objective": -1.0,
            "primal_violation": 0.0,
            "dual_violation": 0.0,
            "complementarity": 0.0,
        }
    )
    assert nnls_optimality_metrics(matrix, vector, np.zeros(2))["dual_violation"] > 0.0
    assert nnls_optimality_metrics(matrix, vector, np.array([1.0, -0.1]))["primal_violation"] > 0.0


def test_relative_l2_error_supports_vectors_and_matrices():
    assert relative_l2_error([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert relative_l2_error(np.eye(2), 2.0 * np.eye(2)) == pytest.approx(0.5)
    assert relative_l2_error([1.0], [1.0, 2.0]) == float("inf")


def test_stable_ellipse_uses_axis_aligned_frame_for_isotropic_covariance():
    border = np.array(((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, -1.0)))
    parameters = stable_ellipse_parameters_from_border(border)
    assert parameters["near_isotropic"] is True
    assert parameters["phi"] == 0.0
    assert parameters["a"] == pytest.approx(parameters["b"])


def test_stable_ellipse_retains_pca_direction_for_anisotropic_covariance():
    border = np.array(((-2.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, -1.0)))
    parameters = stable_ellipse_parameters_from_border(border)
    assert parameters["near_isotropic"] is False
    assert abs(parameters["phi"]) == pytest.approx(np.pi / 2.0)
    assert parameters["a"] > parameters["b"]


def test_conditioning_and_structural_helpers_use_physical_scales():
    assert floor_fraction(1.0e-3, (1.0, 2.0, 3.0)) == pytest.approx(5.0e-4)
    rows = [
        LikelihoodProbeRow(0.7, "axis_ratio", "numpy", -5.0, metadata={"axis_ratio": 0.7}),
        LikelihoodProbeRow(0.7, "axis_ratio", "numpy", -2.0, metadata={"axis_ratio": 0.7}),
        LikelihoodProbeRow(1.0, "axis_ratio", "numpy", -3.0, metadata={"axis_ratio": 1.0}),
        LikelihoodProbeRow(1.0, "axis_ratio", "numpy", -3.0, metadata={"axis_ratio": 1.0}),
    ]
    assert orientation_spans(rows) == {0.7: 3.0, 1.0: 0.0}


def _conditioning_row(noise_scale, policy, floor, curvature, reconstruction, fom):
    return LikelihoodProbeRow(
        parameter=0.9,
        parameter_name="einstein_radius",
        backend="numpy",
        figure_of_merit=fom,
        reconstruction=tuple(reconstruction),
        curvature_diagonal=tuple(curvature),
        conditioned_curvature_diagonal=tuple(curvature),
        noise_scale=noise_scale,
        metadata={
            "curvature_floor_policy": policy,
            "curvature_floor_value": floor,
        },
    )


def test_scale_aware_floor_is_calibrated_from_unfloored_rows():
    controls = [
        _conditioning_row(0.5, "none", 0.0, (4.0, 8.0), (1.0,), -10.0),
        _conditioning_row(1.0, "none", 0.0, (1.0, 2.0), (1.0,), -10.0),
        _conditioning_row(2.0, "none", 0.0, (0.25, 0.5), (1.0,), -10.0),
    ]
    fraction, values = calibrated_scale_aware_floors(controls, configured_floor=1.5e-3)
    assert fraction == pytest.approx(1.0e-3)
    assert values == pytest.approx({0.5: 6.0e-3, 1.0: 1.5e-3, 2.0: 3.75e-4})


def test_conditioning_policies_share_unfloored_denominator_and_report_errors():
    rows = []
    for noise_scale, curvature in ((1.0, (1.0, 2.0)), (2.0, (0.25, 0.5))):
        rows.extend(
            (
                _conditioning_row(noise_scale, "none", 0.0, curvature, (1.0, 2.0), -10.0),
                _conditioning_row(noise_scale, "absolute", 1.5e-3, curvature, (1.0, 2.0), -10.0),
                _conditioning_row(
                    noise_scale,
                    "scale_aware",
                    1.0e-3 * float(np.median(curvature)),
                    curvature,
                    (1.0, 2.1),
                    -9.0,
                ),
            )
        )
    metrics = conditioning_policy_metrics(rows)
    assert metrics["absolute"]["floor_fraction"] == pytest.approx([1.0e-3, 4.0e-3])
    assert metrics["scale_aware"]["floor_fraction"] == pytest.approx([1.0e-3, 1.0e-3])
    assert metrics["scale_aware"]["figure_of_merit_relative_error"] == pytest.approx([0.1, 0.1])
