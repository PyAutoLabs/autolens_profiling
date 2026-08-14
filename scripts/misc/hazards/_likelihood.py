"""Pure analysis helpers shared by likelihood-tier hazard cells and checks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

DEFAULT_ELL_COMPS_SIGMA = 0.3


@dataclass(frozen=True)
class LikelihoodProbeRow:
    """One complete-likelihood evaluation and its inversion diagnostics."""

    parameter: float
    parameter_name: str
    backend: str
    figure_of_merit: float
    reconstruction: tuple[float, ...] = ()
    curvature_diagonal: tuple[float, ...] = ()
    conditioned_curvature_diagonal: tuple[float, ...] = ()
    regularization_diagonal: tuple[float, ...] = ()
    noise_scale: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SolverDiagnosticRow:
    """One solver policy graded on one fixed NNLS system."""

    parameter: float
    parameter_hex: str
    system_backend: str
    solver_policy: str
    reconstruction: tuple[float, ...]
    support: tuple[bool, ...]
    objective: float
    primal_violation: float
    dual_violation: float
    complementarity: float
    reconstruction_relative_error_to_numpy_solver: float
    objective_relative_gap_to_numpy_solver: float
    system_matrix_relative_error_to_numpy: float
    system_data_vector_relative_error_to_numpy: float
    native_fit_reconstruction_relative_error_to_numpy: float
    native_fit_figure_of_merit_relative_error_to_numpy: float
    native_fit_support: tuple[bool, ...]
    numpy_fit_support: tuple[bool, ...]


@dataclass(frozen=True)
class BorderRelocatorComparisonRow:
    """NumPy/JAX comparison through the border-relocation pipeline."""

    parameter: float
    parameter_hex: str
    use_border_relocator: bool
    raw_source_grid_relative_error: float
    relocated_source_grid_relative_error: float
    source_mesh_grid_relative_error: float
    mapping_matrix_relative_error: float
    curvature_reg_matrix_relative_error: float
    data_vector_relative_error: float
    reconstruction_relative_error: float
    figure_of_merit_relative_error: float
    supports_equal: bool
    first_divergent_stage: str | None
    numpy_pca_axes: tuple[float, float]
    jax_pca_axes: tuple[float, float]
    numpy_pca_phi: float
    jax_pca_phi: float
    numpy_pca_relative_eigenvalue_gap: float
    jax_pca_relative_eigenvalue_gap: float
    stable_relocated_source_grid_relative_error: float
    stable_numpy_axes: tuple[float, float]
    stable_jax_axes: tuple[float, float]
    raw_numpy_source_grid: tuple[tuple[float, float], ...]
    raw_jax_source_grid: tuple[tuple[float, float], ...]
    relocated_numpy_source_grid: tuple[tuple[float, float], ...]
    relocated_jax_source_grid: tuple[tuple[float, float], ...]


def ell_comps_radius_from_axis_ratio(axis_ratio: float) -> float:
    """Return the Cartesian ellipticity radius corresponding to an axis ratio."""

    if not 0.0 < axis_ratio <= 1.0:
        raise ValueError("axis ratio must be in (0, 1]")
    return (1.0 - axis_ratio) / (1.0 + axis_ratio)


def isotropic_gaussian_disk_mass(radius: float, *, sigma: float) -> float:
    """Return the mass inside a disk for two independent zero-mean Gaussians."""

    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return float(-np.expm1(-(radius * radius) / (2.0 * sigma * sigma)))


def orientation_degeneracy_persists(sampled_parameters: tuple[str, ...]) -> bool:
    """Return whether axis ratio and angle are independent sampled coordinates."""

    return {"axis_ratio", "angle"}.issubset(sampled_parameters)


def nonfinite_gradient_site_persists(
    origin_gradient: tuple[float, ...],
    neighbourhood_gradients: tuple[tuple[float, ...], ...],
) -> bool:
    """Return whether only the exact site has a non-finite autodiff result."""

    origin = np.asarray(origin_gradient, dtype=float)
    neighbourhood = np.asarray(neighbourhood_gradients, dtype=float)
    return bool(
        origin.size
        and neighbourhood.size
        and np.any(~np.isfinite(origin))
        and np.all(np.isfinite(neighbourhood))
    )


def support_mask(
    values: tuple[float, ...], *, relative_tolerance: float = 1.0e-8
) -> tuple[bool, ...]:
    """Return the numerical NNLS support without baking in an absolute flux scale."""

    array = np.asarray(values, dtype=float)
    if not array.size:
        return ()
    threshold = max(float(np.max(np.abs(array))) * relative_tolerance, np.finfo(float).eps)
    return tuple(bool(value > threshold) for value in array)


def relative_l2_error(candidate, reference) -> float:
    """Scale-free L2 error for vectors or matrices with matching shapes."""

    candidate_array = np.asarray(candidate, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if candidate_array.shape != reference_array.shape:
        return float("inf")
    return float(
        np.linalg.norm(candidate_array - reference_array)
        / max(float(np.linalg.norm(reference_array)), 1.0e-30)
    )


def nnls_optimality_metrics(curvature_reg_matrix, data_vector, reconstruction) -> dict[str, float]:
    """Return scale-normalized objective and KKT residuals for an NNLS solve.

    For ``min 0.5 x.T Q x - q.T x`` subject to ``x >= 0``, optimality
    requires non-negative ``x`` and gradient ``Qx-q`` plus zero
    complementarity ``x * (Qx-q)``.
    """

    matrix = np.asarray(curvature_reg_matrix, dtype=float)
    vector = np.asarray(data_vector, dtype=float)
    solution = np.asarray(reconstruction, dtype=float)
    if matrix.shape != (solution.size, solution.size):
        raise ValueError("curvature-regularization matrix shape does not match solution")
    if vector.shape != solution.shape:
        raise ValueError("data vector shape does not match solution")

    gradient = matrix @ solution - vector
    solution_scale = max(float(np.linalg.norm(solution, ord=np.inf)), 1.0e-30)
    vector_scale = max(float(np.linalg.norm(vector, ord=np.inf)), 1.0e-30)
    return {
        "objective": float(0.5 * solution @ matrix @ solution - vector @ solution),
        "primal_violation": max(0.0, float(-np.min(solution))) / solution_scale,
        "dual_violation": max(0.0, float(-np.min(gradient))) / vector_scale,
        "complementarity": float(np.linalg.norm(solution * gradient, ord=np.inf))
        / (solution_scale * vector_scale),
    }


def stable_ellipse_parameters_from_border(
    border_grid, *, relative_tolerance: float | None = None, eps: float = 1.0e-12
) -> dict[str, float | tuple[float, float]]:
    """PCA ellipse parameters with a deterministic near-isotropic branch.

    Eigenvectors are undefined when covariance eigenvalues are equal. This
    counterfactual uses an axis-aligned frame whenever their relative gap is no
    larger than ``sqrt(machine epsilon)``; otherwise it retains the PCA major
    axis.
    """

    grid = np.asarray(border_grid, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or grid.shape[0] < 2:
        raise ValueError("border grid must have shape (N, 2) with N >= 2")
    tolerance = (
        float(np.sqrt(np.finfo(float).eps))
        if relative_tolerance is None
        else float(relative_tolerance)
    )
    origin = np.mean(grid, axis=0)
    dy = grid[:, 0] - origin[0]
    dx = grid[:, 1] - origin[1]
    coordinates = np.stack((dx, dy), axis=1)
    covariance = coordinates.T @ coordinates / max(coordinates.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    relative_gap = float((eigenvalues[-1] - eigenvalues[0]) / eigenvalue_scale)
    if relative_gap <= tolerance:
        phi = 0.0
    else:
        major = eigenvectors[:, -1]
        phi = float(np.arctan2(major[1], major[0]))
    cosine = np.cos(phi)
    sine = np.sin(phi)
    xprime = cosine * dx + sine * dy
    yprime = -sine * dx + cosine * dy
    return {
        "origin": (float(origin[0]), float(origin[1])),
        "a": float(np.max(np.abs(xprime)) + eps),
        "b": float(np.max(np.abs(yprime)) + eps),
        "phi": phi,
        "relative_eigenvalue_gap": relative_gap,
        "relative_tolerance": tolerance,
        "near_isotropic": bool(relative_gap <= tolerance),
    }


def support_transition_locations(rows: list[LikelihoodProbeRow]) -> list[float]:
    """Midpoints where adjacent evaluations change active NNLS support."""

    ordered = sorted(rows, key=lambda row: row.parameter)
    locations: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        if support_mask(left.reconstruction) != support_mask(right.reconstruction):
            locations.append(0.5 * (left.parameter + right.parameter))
    return locations


def epsilon_neighbourhood_mass(
    centres: list[float], *, epsilon: float, lower: float, upper: float
) -> float:
    """Uniform prior mass of the union of clipped epsilon-neighbourhoods."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    intervals = sorted(
        (max(lower, centre - epsilon), min(upper, centre + epsilon))
        for centre in centres
        if lower - epsilon <= centre <= upper + epsilon
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged) / (upper - lower)


def diagonal_scale(diagonal: tuple[float, ...]) -> float:
    """Robust positive scale for a matrix represented by its diagonal."""

    values = np.abs(np.asarray(diagonal, dtype=float))
    positive = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(positive)) if positive.size else 0.0


def floor_fraction(floor: float, diagonal: tuple[float, ...]) -> float:
    scale = diagonal_scale(diagonal)
    return float(floor / scale) if scale else float("inf")


def conditioned_diagonal(row: LikelihoodProbeRow) -> tuple[float, ...]:
    """Diagonal entries actually touched by the curvature-floor policy."""

    return row.conditioned_curvature_diagonal or row.curvature_diagonal


def calibrated_scale_aware_floors(
    control_rows: list[LikelihoodProbeRow],
    *,
    configured_floor: float,
    reference_noise_scale: float = 1.0,
) -> tuple[float, dict[float, float]]:
    """Calibrate a relative floor against one unfloored likelihood row.

    The returned fraction makes the scale-aware floor equal the configured
    absolute floor at ``reference_noise_scale``. Every candidate value is
    derived from an unfloored curvature diagonal, so neither policy defines
    its own denominator.
    """

    rows_by_noise = {row.noise_scale: row for row in control_rows}
    if len(rows_by_noise) != len(control_rows):
        raise ValueError("control rows must have unique noise scales")
    try:
        reference = rows_by_noise[reference_noise_scale]
    except KeyError as error:
        raise ValueError("reference noise scale is missing from control rows") from error
    fraction = floor_fraction(configured_floor, conditioned_diagonal(reference))
    if not np.isfinite(fraction):
        raise ValueError("reference curvature diagonal has no finite positive scale")
    values = {
        noise_scale: fraction * diagonal_scale(conditioned_diagonal(row))
        for noise_scale, row in sorted(rows_by_noise.items())
    }
    return fraction, values


def conditioning_policy_metrics(
    rows: list[LikelihoodProbeRow], *, reference_policy: str = "absolute"
) -> dict[str, dict[str, list[float]]]:
    """Compare conditioning policies against the same unfloored scale."""

    grouped: dict[tuple[str, float], LikelihoodProbeRow] = {}
    for row in rows:
        policy = str(row.metadata["curvature_floor_policy"])
        key = (policy, row.noise_scale)
        if key in grouped:
            raise ValueError(f"duplicate conditioning row for {key}")
        grouped[key] = row

    noise_scales = sorted(
        noise_scale for policy, noise_scale in grouped if policy == reference_policy
    )
    if not noise_scales:
        raise ValueError(f"reference policy {reference_policy!r} is missing")
    controls = {noise_scale: grouped.get(("none", noise_scale)) for noise_scale in noise_scales}
    if any(row is None for row in controls.values()):
        raise ValueError("an unfloored control is required at every noise scale")

    metrics: dict[str, dict[str, list[float]]] = {}
    policies = sorted({policy for policy, _ in grouped})
    for policy in policies:
        policy_metrics = {
            "noise_scale": [],
            "floor_value": [],
            "floor_fraction": [],
            "figure_of_merit_relative_error": [],
            "reconstruction_relative_error": [],
        }
        for noise_scale in noise_scales:
            row = grouped.get((policy, noise_scale))
            reference = grouped[(reference_policy, noise_scale)]
            control = controls[noise_scale]
            if row is None or control is None:
                raise ValueError(f"policy {policy!r} is incomplete")
            floor = float(row.metadata["curvature_floor_value"])
            reference_scale = max(abs(reference.figure_of_merit), 1.0)
            expected = np.asarray(reference.reconstruction, dtype=float)
            candidate = np.asarray(row.reconstruction, dtype=float)
            if candidate.shape != expected.shape:
                reconstruction_error = float("inf")
            else:
                reconstruction_error = float(
                    np.linalg.norm(candidate - expected)
                    / max(float(np.linalg.norm(expected)), 1.0e-14)
                )
            policy_metrics["noise_scale"].append(float(noise_scale))
            policy_metrics["floor_value"].append(floor)
            policy_metrics["floor_fraction"].append(
                floor_fraction(floor, conditioned_diagonal(control))
            )
            policy_metrics["figure_of_merit_relative_error"].append(
                abs(row.figure_of_merit - reference.figure_of_merit) / reference_scale
            )
            policy_metrics["reconstruction_relative_error"].append(reconstruction_error)
        metrics[policy] = policy_metrics
    return metrics


def backend_error_curves(
    rows: list[LikelihoodProbeRow], *, reference_backend: str = "numpy"
) -> dict[str, dict[str, list[float]]]:
    """Align backends by parameter and report FoM and reconstruction errors."""

    reference = {
        (row.parameter_name, row.parameter, row.noise_scale): row
        for row in rows
        if row.backend == reference_backend
    }
    curves: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.backend == reference_backend:
            continue
        key = (row.parameter_name, row.parameter, row.noise_scale)
        target = reference.get(key)
        if target is None:
            continue
        fom_scale = max(abs(target.figure_of_merit), 1.0)
        fom_error = abs(row.figure_of_merit - target.figure_of_merit) / fom_scale
        candidate = np.asarray(row.reconstruction, dtype=float)
        expected = np.asarray(target.reconstruction, dtype=float)
        if candidate.shape != expected.shape:
            reconstruction_error = float("inf")
        else:
            reconstruction_error = float(
                np.linalg.norm(candidate - expected) / max(float(np.linalg.norm(expected)), 1.0e-14)
            )
        curve = curves.setdefault(
            row.backend,
            {"parameter": [], "figure_of_merit": [], "reconstruction": []},
        )
        curve["parameter"].append(row.parameter)
        curve["figure_of_merit"].append(float(fom_error))
        curve["reconstruction"].append(reconstruction_error)
    return curves


def backend_divergence_persists(
    curves: dict[str, dict[str, list[float]]],
    *,
    relative_tolerance: float = 1.0e-8,
) -> bool:
    """Return whether any full-likelihood backend error exceeds parity noise."""

    if relative_tolerance <= 0.0:
        raise ValueError("relative tolerance must be positive")
    for curve in curves.values():
        for quantity in ("figure_of_merit", "reconstruction"):
            values = np.asarray(curve.get(quantity, ()), dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values > relative_tolerance):
                return True
    return False


def orientation_spans(rows: list[LikelihoodProbeRow]) -> dict[float, float]:
    """Figure-of-merit span over orientation at every axis ratio."""

    grouped: dict[float, list[float]] = {}
    for row in rows:
        axis_ratio = float(row.metadata["axis_ratio"])
        grouped.setdefault(axis_ratio, []).append(row.figure_of_merit)
    return {
        axis_ratio: float(max(values) - min(values))
        for axis_ratio, values in sorted(grouped.items())
    }


def _load_cell(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("autolens_hazard_pixelization", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load hazard cell: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imaging_pixelization_probe(context) -> dict[str, list[LikelihoodProbeRow]]:
    """Load and cache the small imaging cell; four checks share this one run."""

    key = "imaging_pixelization_probe"
    cached = context.cache.get(key)
    if cached is None:
        path = context.repo_root / "scripts" / "imaging" / "hazards" / "pixelization.py"
        cached = _load_cell(path).run_probe(backends=context.backends)
        context.cache[key] = cached
    return cached
