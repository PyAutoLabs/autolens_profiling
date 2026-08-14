"""Compare the algorithmically distinct NumPy and JAX positive solvers."""

from __future__ import annotations

import numpy as np

from hazards._anchor import maybe_anchor_from_pattern
from hazards._likelihood import (
    backend_divergence_persists,
    backend_error_curves,
    imaging_pixelization_probe,
)
from hazards._measure import Measurement, reachability_measurement
from hazards._record import Finding
from hazards.checks._base import HazardCheck, ScanContext


class SolverDivergenceCheck(HazardCheck):
    name = "solver_divergence"
    subject = "likelihood"

    def run(self, context: ScanContext) -> list[Finding]:
        rows = [
            row
            for row in imaging_pixelization_probe(context)["inversion"]
            if row.noise_scale == 1.0
        ]
        curves = backend_error_curves(rows)
        if not curves:
            return []
        native_maximum = max(
            max(curve[quantity])
            for curve in curves.values()
            for quantity in ("figure_of_merit", "reconstruction")
        )
        if not backend_divergence_persists(curves):
            return []
        diagnostic = imaging_pixelization_probe(context)["solver_diagnostic"]
        policy_maxima = {
            policy: {
                "reconstruction_relative_error_to_numpy_solver": max(
                    row.reconstruction_relative_error_to_numpy_solver
                    for row in diagnostic
                    if row.solver_policy == policy
                ),
                "objective_relative_gap_to_numpy_solver": max(
                    abs(row.objective_relative_gap_to_numpy_solver)
                    for row in diagnostic
                    if row.solver_policy == policy
                ),
                "complementarity": max(
                    row.complementarity for row in diagnostic if row.solver_policy == policy
                ),
            }
            for policy in ("jax_default", "jax_tight", "jax_relaxed")
        }
        system_matrix_maximum = max(row.system_matrix_relative_error_to_numpy for row in diagnostic)
        system_vector_maximum = max(
            row.system_data_vector_relative_error_to_numpy for row in diagnostic
        )
        native_reconstruction_maximum = max(
            row.native_fit_reconstruction_relative_error_to_numpy for row in diagnostic
        )
        support_boundary = [
            {
                "parameter": row.parameter,
                "parameter_hex": row.parameter_hex,
                "system_backend": row.system_backend,
                "native_fit_support": list(row.native_fit_support),
                "numpy_fit_support": list(row.numpy_fit_support),
                "system_matrix_relative_error_to_numpy": (
                    row.system_matrix_relative_error_to_numpy
                ),
            }
            for row in diagnostic
            if row.solver_policy == "numpy_active_set"
        ]
        relocation = imaging_pixelization_probe(context)["border_relocator"]
        relocated = [row for row in relocation if row.use_border_relocator]
        unrelocated = [row for row in relocation if not row.use_border_relocator]
        relocation_maxima = {
            field: max(getattr(row, field) for row in relocated)
            for field in (
                "raw_source_grid_relative_error",
                "relocated_source_grid_relative_error",
                "source_mesh_grid_relative_error",
                "mapping_matrix_relative_error",
                "curvature_reg_matrix_relative_error",
                "data_vector_relative_error",
                "reconstruction_relative_error",
                "figure_of_merit_relative_error",
                "stable_relocated_source_grid_relative_error",
            )
        }
        disabled_maxima = {
            field: max(getattr(row, field) for row in unrelocated)
            for field in (
                "curvature_reg_matrix_relative_error",
                "reconstruction_relative_error",
                "figure_of_merit_relative_error",
            )
        }
        pca_records = [
            {
                "parameter": row.parameter,
                "parameter_hex": row.parameter_hex,
                "first_divergent_stage": row.first_divergent_stage,
                "numpy_pca_axes": list(row.numpy_pca_axes),
                "jax_pca_axes": list(row.jax_pca_axes),
                "numpy_pca_phi": row.numpy_pca_phi,
                "jax_pca_phi": row.jax_pca_phi,
                "numpy_pca_relative_eigenvalue_gap": (row.numpy_pca_relative_eigenvalue_gap),
                "jax_pca_relative_eigenvalue_gap": row.jax_pca_relative_eigenvalue_gap,
                "stable_numpy_axes": list(row.stable_numpy_axes),
                "stable_jax_axes": list(row.stable_jax_axes),
            }
            for row in relocated
        ]
        worst_relocation = max(relocated, key=lambda row: row.relocated_source_grid_relative_error)
        worst_point_coordinates = {
            "parameter": worst_relocation.parameter,
            "parameter_hex": worst_relocation.parameter_hex,
            "raw_numpy_source_grid": [
                list(value) for value in worst_relocation.raw_numpy_source_grid
            ],
            "raw_jax_source_grid": [list(value) for value in worst_relocation.raw_jax_source_grid],
            "relocated_numpy_source_grid": [
                list(value) for value in worst_relocation.relocated_numpy_source_grid
            ],
            "relocated_jax_source_grid": [
                list(value) for value in worst_relocation.relocated_jax_source_grid
            ],
        }
        numpy_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/util/fnnls.py",
            pattern="def fnnls_cholesky(",
            after=18,
            symbol="autoarray.util.fnnls.fnnls_cholesky",
        )
        jax_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/util/jax_nnls.py",
            pattern="def solve_nnls_primal(",
            after=18,
            symbol="autoarray.util.jax_nnls.solve_nnls_primal",
        )
        dispatch_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/inversion/inversion/inversion_util.py",
            pattern='if xp.__name__.startswith("jax"):',
            after=28,
            symbol="autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from",
        )
        relocator_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoArray",
            path="autoarray/inversion/mesh/border_relocator.py",
            pattern="def ellipse_params_via_border_pca_from(",
            after=45,
            symbol="autoarray.inversion.mesh.border_relocator.ellipse_params_via_border_pca_from",
        )
        return [
            Finding(
                finding_id="likelihood.imaging-pixelization.positive-solver-backend-divergence",
                title="Degenerate border PCA drives the measured backend divergence",
                summary=(
                    "The first native-path difference is border PCA relocation: raw source "
                    f"grids agree to {relocation_maxima['raw_source_grid_relative_error']:.3e}, "
                    "but near-equal covariance eigenvalues select backend-dependent axes and "
                    f"produce {relocation_maxima['relocated_source_grid_relative_error']:.3e} "
                    "relocated-grid error. Disabling relocation restores the curvature "
                    f"system to {disabled_maxima['curvature_reg_matrix_relative_error']:.3e}."
                ),
                hazard_class="solver_divergence",
                tier=2,
                subject="likelihood",
                subject_name="imaging_pixelization",
                backends=("numpy", "jax"),
                measurements=(
                    Measurement(
                        basis="error_curve",
                        value=native_maximum,
                        unit="relative_error",
                        details={"reference": "numpy_fnnls", "curves": curves},
                    ),
                    Measurement(
                        basis="error_curve",
                        value=policy_maxima["jax_default"][
                            "reconstruction_relative_error_to_numpy_solver"
                        ],
                        unit="same_system_solver_relative_error",
                        details={"policy_maxima": policy_maxima},
                    ),
                    Measurement(
                        basis="error_curve",
                        value=system_matrix_maximum,
                        unit="backend_system_relative_error",
                        details={
                            "curvature_regularization_matrix": system_matrix_maximum,
                            "data_vector": system_vector_maximum,
                            "ulp_neighbourhood": support_boundary,
                        },
                    ),
                    Measurement(
                        basis="error_curve",
                        value=relocation_maxima["relocated_source_grid_relative_error"],
                        unit="border_relocator_backend_relative_error",
                        details={
                            "enabled_maxima": relocation_maxima,
                            "disabled_maxima": disabled_maxima,
                            "first_divergent_stage": "border_pca_relocation",
                            "pca_records": pca_records,
                            "worst_point_coordinates": worst_point_coordinates,
                        },
                    ),
                    reachability_measurement(
                        reachable_via=(
                            "FitImaging.numpy.active-set-fnnls",
                            "FitImaging.jax.pdip-jacobi",
                        )
                    ),
                ),
                anchors=tuple(
                    anchor
                    for anchor in (
                        numpy_anchor,
                        jax_anchor,
                        dispatch_anchor,
                        relocator_anchor,
                    )
                    if anchor is not None
                ),
                code_exists=True,
                reachable_via=(
                    "FitImaging.numpy.active-set-fnnls",
                    "FitImaging.jax.pdip-jacobi",
                ),
                blocked_by=(),
                affects_science=None,
                backend_reachability={
                    "numpy": {
                        "algorithm": "active-set FNNLS",
                        "same_system_diagnosis": "agrees with JAX",
                    },
                    "jax": {
                        "algorithm": "PDIP with Jacobi preconditioning",
                        "same_system_diagnosis": "agrees with NumPy",
                    },
                },
                reproducer={
                    "parameter": "einstein_radius",
                    "curves": curves,
                    "same_system_reconstruction_error_max": {
                        policy: values["reconstruction_relative_error_to_numpy_solver"]
                        for policy, values in policy_maxima.items()
                    },
                    "system_matrix_relative_error_max": system_matrix_maximum,
                    "system_data_vector_relative_error_max": system_vector_maximum,
                    "native_fit_reconstruction_relative_error_max": (native_reconstruction_maximum),
                    "ulp_neighbourhood": support_boundary,
                    "border_relocator": {
                        "enabled_maxima": relocation_maxima,
                        "disabled_maxima": disabled_maxima,
                        "pca_records": pca_records,
                        "worst_point_coordinates": worst_point_coordinates,
                        "near_isotropic_tolerance": float(np.sqrt(np.finfo(float).eps)),
                    },
                    "recommendation": (
                        "Open a bounded PyAutoArray border-relocator parity task: when PCA "
                        "eigenvalues are equal within a scale-aware tolerance, select a "
                        "deterministic axis before deriving ellipse extents. Keep solver "
                        "defaults unchanged."
                    ),
                },
            )
        ]
