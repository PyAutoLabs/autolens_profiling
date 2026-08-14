"""Small complete-likelihood fixture for tier-2 imaging hazard checks.

The cell intentionally uses a 7x7 image and 3x3 rectangular source mesh.  It is
large enough to exercise the real ``FitImaging`` inversion while remaining a
diagnostic, not a runtime benchmark.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path

import autoarray as aa
import autolens as al
import numpy as np


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
MISC_ROOT = REPO_ROOT / "scripts" / "misc"
if str(MISC_ROOT) not in sys.path:
    sys.path.insert(0, str(MISC_ROOT))

from hazards._likelihood import (  # noqa: E402
    BorderRelocatorComparisonRow,
    LikelihoodProbeRow,
    SolverDiagnosticRow,
    calibrated_scale_aware_floors,
    nnls_optimality_metrics,
    relative_l2_error,
    stable_ellipse_parameters_from_border,
    support_mask,
)

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)


@lru_cache(maxsize=3)
def _dataset(noise_scale: float = 1.0):
    data_native = np.zeros((7, 7))
    data_native[2:5, 2:5] = np.asarray(((0.1, 1.0, 0.0), (0.2, 3.0, 0.3), (0.0, 0.5, 0.1)))
    data = aa.Array2D.no_mask(values=data_native, pixel_scales=(1.0, 1.0))
    noise_map = aa.Array2D.full(
        fill_value=2.0 * noise_scale,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
    )
    kernel = aa.Array2D.no_mask(
        values=np.asarray(((0.0, 0.5, 0.0), (0.5, 1.0, 0.5), (0.0, 0.5, 0.0))),
        pixel_scales=(1.0, 1.0),
    )
    mask = aa.Mask2D(
        mask=np.asarray(
            (
                (True, True, True, True, True, True, True),
                (True, True, True, True, True, True, True),
                (True, True, False, False, False, True, True),
                (True, True, False, False, False, True, True),
                (True, True, False, False, False, True, True),
                (True, True, True, True, True, True, True),
                (True, True, True, True, True, True, True),
            )
        ),
        pixel_scales=(1.0, 1.0),
    )
    return aa.Imaging(
        data=data,
        psf=aa.Convolver(kernel=kernel),
        noise_map=noise_map,
        over_sample_size_lp=1,
    ).apply_mask(mask=mask)


def _array_module(backend: str):
    if backend == "numpy":
        return np
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        return jnp
    raise ValueError(f"unsupported backend: {backend}")


def _inversion_evaluation(
    *,
    backend: str,
    einstein_radius: float,
    noise_scale: float,
    curvature_floor: float | None = None,
    curvature_floor_policy: str = "absolute",
    nnls_solver_tol: float | None = None,
    nnls_max_iter: int | None = None,
    use_border_relocator: bool | None = None,
) -> tuple[LikelihoodProbeRow, np.ndarray, np.ndarray, dict]:
    xp = _array_module(backend)
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    tracer = al.Tracer(
        galaxies=(
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp_linear.SersicSph(
                    effective_radius=0.6,
                    sersic_index=2.0,
                ),
                mass=al.mp.IsothermalSph(einstein_radius=einstein_radius),
            ),
            al.Galaxy(redshift=1.0, pixelization=pixelization),
        )
    )
    settings = al.Settings(
        use_positive_only_solver=True,
        use_edge_zeroed_pixels=False,
        no_regularization_add_to_curvature_diag_value=curvature_floor,
        nnls_solver_tol=nnls_solver_tol,
        nnls_max_iter=nnls_max_iter,
        use_border_relocator=use_border_relocator,
    )
    applied_floor = float(settings.no_regularization_add_to_curvature_diag_value)
    fit = al.FitImaging(
        dataset=_dataset(noise_scale=noise_scale),
        tracer=tracer,
        settings=settings,
        xp=xp,
    )
    inversion = fit.inversion
    curvature_diagonal = np.diag(np.asarray(inversion.curvature_matrix, dtype=float))
    conditioned_indices = tuple(int(index) for index in inversion.no_regularization_index_list)
    row = LikelihoodProbeRow(
        parameter=float(einstein_radius),
        parameter_name="einstein_radius",
        backend=backend,
        figure_of_merit=float(np.asarray(fit.figure_of_merit)),
        reconstruction=tuple(np.asarray(inversion.reconstruction, dtype=float).tolist()),
        curvature_diagonal=tuple(curvature_diagonal.tolist()),
        conditioned_curvature_diagonal=tuple(
            curvature_diagonal[list(conditioned_indices)].tolist()
        ),
        regularization_diagonal=tuple(
            np.diag(np.asarray(inversion.regularization_matrix, dtype=float)).tolist()
        ),
        noise_scale=float(noise_scale),
        metadata={
            "curvature_floor_policy": curvature_floor_policy,
            "curvature_floor_value": applied_floor,
            "conditioned_indices": list(conditioned_indices),
            "nnls_solver_tol": nnls_solver_tol,
            "nnls_max_iter": nnls_max_iter,
        },
    )
    mapper = inversion.linear_obj_list[-1]
    raw_source_grid = np.asarray(
        fit.tracer_to_inversion.traced_grid_2d_list_of_inversion[-1], dtype=float
    )
    border_indexes = np.asarray(fit.dataset.mask.derive_indexes.border_slim, dtype=int)
    from autoarray.inversion.mesh.border_relocator import (
        ellipse_params_via_border_pca_from,
    )

    pca_origin, pca_a, pca_b, pca_phi = ellipse_params_via_border_pca_from(
        xp.asarray(raw_source_grid[border_indexes]), xp=xp
    )
    details = {
        "raw_source_grid": raw_source_grid,
        "relocated_source_grid": np.asarray(mapper.source_plane_data_grid, dtype=float),
        "source_mesh_grid": np.asarray(mapper.source_plane_mesh_grid, dtype=float),
        "mapping_matrix": np.asarray(mapper.mapping_matrix, dtype=float),
        "pca_origin": tuple(np.asarray(pca_origin, dtype=float).tolist()),
        "pca_axes": (float(np.asarray(pca_a)), float(np.asarray(pca_b))),
        "pca_phi": float(np.asarray(pca_phi)),
        "border_indexes": border_indexes,
    }
    return (
        row,
        np.asarray(inversion.curvature_reg_matrix, dtype=float),
        np.asarray(inversion.data_vector, dtype=float),
        details,
    )


def _inversion_row(**kwargs) -> LikelihoodProbeRow:
    return _inversion_evaluation(**kwargs)[0]


def _solve_nnls_system(
    curvature_reg_matrix: np.ndarray,
    data_vector: np.ndarray,
    *,
    solver_policy: str,
) -> np.ndarray:
    from autoarray.inversion.inversion.inversion_util import (
        reconstruction_positive_only_from,
    )

    policies = {
        "numpy_active_set": ("numpy", None, None),
        "jax_default": ("jax", None, None),
        "jax_tight": ("jax", 1.0e-14, 200),
        "jax_relaxed": ("jax", 1.0e-6, 50),
    }
    backend, solver_tol, max_iter = policies[solver_policy]
    xp = _array_module(backend)
    settings = al.Settings(nnls_solver_tol=solver_tol, nnls_max_iter=max_iter)
    return np.asarray(
        reconstruction_positive_only_from(
            data_vector=xp.asarray(data_vector),
            curvature_reg_matrix=xp.asarray(curvature_reg_matrix),
            settings=settings,
            xp=xp,
        ),
        dtype=float,
    )


def _solver_diagnostic_rows() -> list[SolverDiagnosticRow]:
    """Separate solver effects from backend system-construction effects."""

    centre = 1.55
    radii = (
        float(np.nextafter(centre, -np.inf)),
        centre,
        float(np.nextafter(centre, np.inf)),
    )
    rows: list[SolverDiagnosticRow] = []
    for radius in radii:
        evaluations = {
            backend: _inversion_evaluation(
                backend=backend,
                einstein_radius=radius,
                noise_scale=1.0,
            )
            for backend in ("numpy", "jax")
        }
        numpy_fit, numpy_matrix, numpy_vector, _ = evaluations["numpy"]
        for system_backend, (native_fit, matrix, vector, _) in evaluations.items():
            numpy_solution = _solve_nnls_system(
                matrix,
                vector,
                solver_policy="numpy_active_set",
            )
            numpy_metrics = nnls_optimality_metrics(matrix, vector, numpy_solution)
            matrix_error = relative_l2_error(matrix, numpy_matrix)
            vector_error = relative_l2_error(vector, numpy_vector)
            native_reconstruction_error = relative_l2_error(
                native_fit.reconstruction,
                numpy_fit.reconstruction,
            )
            native_fom_error = abs(native_fit.figure_of_merit - numpy_fit.figure_of_merit) / max(
                abs(numpy_fit.figure_of_merit), 1.0
            )
            for solver_policy in (
                "numpy_active_set",
                "jax_default",
                "jax_tight",
                "jax_relaxed",
            ):
                solution = _solve_nnls_system(
                    matrix,
                    vector,
                    solver_policy=solver_policy,
                )
                metrics = nnls_optimality_metrics(matrix, vector, solution)
                rows.append(
                    SolverDiagnosticRow(
                        parameter=radius,
                        parameter_hex=radius.hex(),
                        system_backend=system_backend,
                        solver_policy=solver_policy,
                        reconstruction=tuple(solution.tolist()),
                        support=support_mask(tuple(solution.tolist())),
                        objective=metrics["objective"],
                        primal_violation=metrics["primal_violation"],
                        dual_violation=metrics["dual_violation"],
                        complementarity=metrics["complementarity"],
                        reconstruction_relative_error_to_numpy_solver=relative_l2_error(
                            solution, numpy_solution
                        ),
                        objective_relative_gap_to_numpy_solver=(
                            metrics["objective"] - numpy_metrics["objective"]
                        )
                        / max(abs(numpy_metrics["objective"]), 1.0),
                        system_matrix_relative_error_to_numpy=matrix_error,
                        system_data_vector_relative_error_to_numpy=vector_error,
                        native_fit_reconstruction_relative_error_to_numpy=(
                            native_reconstruction_error
                        ),
                        native_fit_figure_of_merit_relative_error_to_numpy=(native_fom_error),
                        native_fit_support=support_mask(native_fit.reconstruction),
                        numpy_fit_support=support_mask(numpy_fit.reconstruction),
                    )
                )
    return rows


def _relocated_grid_with_stable_ellipse(details: dict) -> tuple[np.ndarray, dict]:
    from autoarray.inversion.mesh.border_relocator import (
        relocated_grid_via_ellipse_border_from,
    )

    raw_grid = details["raw_source_grid"]
    border_grid = raw_grid[details["border_indexes"]]
    parameters = stable_ellipse_parameters_from_border(border_grid)
    relocated = relocated_grid_via_ellipse_border_from(
        grid=raw_grid,
        origin=np.asarray(parameters["origin"]),
        a=parameters["a"],
        b=parameters["b"],
        phi=parameters["phi"],
        xp=np,
    )
    return np.asarray(relocated, dtype=float), parameters


def _border_relocator_comparison_rows() -> list[BorderRelocatorComparisonRow]:
    centre = 1.55
    radii = (
        float(np.nextafter(centre, -np.inf)),
        centre,
        float(np.nextafter(centre, np.inf)),
    )
    rows: list[BorderRelocatorComparisonRow] = []
    for radius in radii:
        for use_border_relocator in (True, False):
            evaluations = {
                backend: _inversion_evaluation(
                    backend=backend,
                    einstein_radius=radius,
                    noise_scale=1.0,
                    use_border_relocator=use_border_relocator,
                )
                for backend in ("numpy", "jax")
            }
            numpy_row, numpy_matrix, numpy_vector, numpy_details = evaluations["numpy"]
            jax_row, jax_matrix, jax_vector, jax_details = evaluations["jax"]
            raw_error = relative_l2_error(
                jax_details["raw_source_grid"], numpy_details["raw_source_grid"]
            )
            relocated_error = relative_l2_error(
                jax_details["relocated_source_grid"],
                numpy_details["relocated_source_grid"],
            )
            mesh_error = relative_l2_error(
                jax_details["source_mesh_grid"], numpy_details["source_mesh_grid"]
            )
            mapping_error = relative_l2_error(
                jax_details["mapping_matrix"], numpy_details["mapping_matrix"]
            )
            matrix_error = relative_l2_error(jax_matrix, numpy_matrix)
            vector_error = relative_l2_error(jax_vector, numpy_vector)
            reconstruction_error = relative_l2_error(
                jax_row.reconstruction, numpy_row.reconstruction
            )
            stable_numpy_grid, stable_numpy = _relocated_grid_with_stable_ellipse(numpy_details)
            stable_jax_grid, stable_jax = _relocated_grid_with_stable_ellipse(jax_details)
            stages = (
                ("traced_source_grid", raw_error),
                ("border_pca_relocation", relocated_error),
                ("source_mesh", mesh_error),
                ("mapping_matrix", mapping_error),
                ("curvature_regularization_matrix", matrix_error),
                ("data_vector", vector_error),
                ("reconstruction", reconstruction_error),
            )
            first_divergent_stage = next(
                (stage for stage, error in stages if error > 1.0e-12), None
            )
            rows.append(
                BorderRelocatorComparisonRow(
                    parameter=radius,
                    parameter_hex=radius.hex(),
                    use_border_relocator=use_border_relocator,
                    raw_source_grid_relative_error=raw_error,
                    relocated_source_grid_relative_error=relocated_error,
                    source_mesh_grid_relative_error=mesh_error,
                    mapping_matrix_relative_error=mapping_error,
                    curvature_reg_matrix_relative_error=matrix_error,
                    data_vector_relative_error=vector_error,
                    reconstruction_relative_error=reconstruction_error,
                    figure_of_merit_relative_error=abs(
                        jax_row.figure_of_merit - numpy_row.figure_of_merit
                    )
                    / max(abs(numpy_row.figure_of_merit), 1.0),
                    supports_equal=(
                        support_mask(jax_row.reconstruction)
                        == support_mask(numpy_row.reconstruction)
                    ),
                    first_divergent_stage=first_divergent_stage,
                    numpy_pca_axes=numpy_details["pca_axes"],
                    jax_pca_axes=jax_details["pca_axes"],
                    numpy_pca_phi=numpy_details["pca_phi"],
                    jax_pca_phi=jax_details["pca_phi"],
                    numpy_pca_relative_eigenvalue_gap=float(
                        stable_numpy["relative_eigenvalue_gap"]
                    ),
                    jax_pca_relative_eigenvalue_gap=float(stable_jax["relative_eigenvalue_gap"]),
                    stable_relocated_source_grid_relative_error=relative_l2_error(
                        stable_jax_grid, stable_numpy_grid
                    ),
                    stable_numpy_axes=(stable_numpy["a"], stable_numpy["b"]),
                    stable_jax_axes=(stable_jax["a"], stable_jax["b"]),
                    raw_numpy_source_grid=tuple(
                        tuple(value) for value in numpy_details["raw_source_grid"].tolist()
                    ),
                    raw_jax_source_grid=tuple(
                        tuple(value) for value in jax_details["raw_source_grid"].tolist()
                    ),
                    relocated_numpy_source_grid=tuple(
                        tuple(value) for value in numpy_details["relocated_source_grid"].tolist()
                    ),
                    relocated_jax_source_grid=tuple(
                        tuple(value) for value in jax_details["relocated_source_grid"].tolist()
                    ),
                )
            )
    return rows


def _structural_row(*, backend: str, axis_ratio: float, angle: float) -> LikelihoodProbeRow:
    xp = _array_module(backend)
    ell_comps = al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle)
    tracer = al.Tracer(
        galaxies=(
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.Sersic(
                    ell_comps=ell_comps,
                    intensity=1.0,
                    effective_radius=1.0,
                    sersic_index=2.0,
                ),
            ),
        )
    )
    fit = al.FitImaging(dataset=_dataset(), tracer=tracer, xp=xp)
    return LikelihoodProbeRow(
        parameter=float(axis_ratio),
        parameter_name="axis_ratio",
        backend=backend,
        figure_of_merit=float(np.asarray(fit.figure_of_merit)),
        metadata={"axis_ratio": float(axis_ratio), "angle": float(angle)},
    )


def _ell_comps_gradient_probe() -> dict:
    """Differentiate a complete off-centre Sersic likelihood at its circular point."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def figure_of_merit(ell_comps):
        tracer = al.Tracer(
            galaxies=(
                al.Galaxy(
                    redshift=0.5,
                    bulge=al.lp.Sersic(
                        centre=(0.13, -0.17),
                        ell_comps=(ell_comps[0], ell_comps[1]),
                        intensity=1.0,
                        effective_radius=1.0,
                        sersic_index=2.0,
                    ),
                ),
            )
        )
        return al.FitImaging(dataset=_dataset(), tracer=tracer, xp=jnp).figure_of_merit

    points = np.asarray(
        (
            (0.0, 0.0),
            (1.0e-8, 0.0),
            (-1.0e-8, 0.0),
            (0.0, 1.0e-8),
            (0.0, -1.0e-8),
            (1.0e-6, 0.0),
            (-1.0e-6, 0.0),
            (0.0, 1.0e-6),
            (0.0, -1.0e-6),
        ),
        dtype=float,
    )
    values = []
    gradients = []
    for point in points:
        point_array = jnp.asarray(point, dtype=jnp.float64)
        values.append(float(np.asarray(figure_of_merit(point_array))))
        gradients.append(tuple(np.asarray(jax.grad(figure_of_merit)(point_array), dtype=float)))
    return {
        "points": [tuple(point) for point in points],
        "figure_of_merit": values,
        "gradients": gradients,
    }


def run_probe(backends: tuple[str, ...] = ("numpy", "jax")) -> dict[str, list]:
    """Evaluate the full likelihood over bounded diagnostic parameter grids."""

    einstein_radii = np.linspace(0.1, 1.6, 31)
    inversion = [
        _inversion_row(
            backend=backend,
            einstein_radius=float(einstein_radius),
            noise_scale=1.0,
        )
        for backend in backends
        for einstein_radius in einstein_radii
    ]
    inversion.extend(
        _inversion_row(backend=backend, einstein_radius=0.9, noise_scale=noise_scale)
        for backend in backends
        for noise_scale in (0.5, 2.0)
    )
    configured_floor = float(al.Settings().no_regularization_add_to_curvature_diag_value)
    control_rows = [
        _inversion_row(
            backend="numpy",
            einstein_radius=0.9,
            noise_scale=noise_scale,
            curvature_floor=0.0,
            curvature_floor_policy="none",
        )
        for noise_scale in (0.5, 1.0, 2.0)
    ]
    _, scale_aware_values = calibrated_scale_aware_floors(
        control_rows,
        configured_floor=configured_floor,
    )
    absolute_rows = [row for row in inversion if row.backend == "numpy" and row.parameter == 0.9]
    scale_aware_rows = [
        _inversion_row(
            backend="numpy",
            einstein_radius=0.9,
            noise_scale=noise_scale,
            curvature_floor=scale_aware_values[noise_scale],
            curvature_floor_policy="scale_aware",
        )
        for noise_scale in (0.5, 1.0, 2.0)
    ]
    conditioning = control_rows + absolute_rows + scale_aware_rows
    structural = [
        _structural_row(backend=backend, axis_ratio=axis_ratio, angle=angle)
        for backend in backends
        for axis_ratio in (0.7, 0.9, 0.99, 1.0)
        for angle in (0.0, 30.0, 60.0, 90.0)
    ]
    return {
        "inversion": inversion,
        "conditioning": conditioning,
        "solver_diagnostic": (
            _solver_diagnostic_rows() if {"numpy", "jax"}.issubset(set(backends)) else []
        ),
        "border_relocator": (
            _border_relocator_comparison_rows() if {"numpy", "jax"}.issubset(set(backends)) else []
        ),
        "structural": structural,
        "ell_comps_gradient": ([_ell_comps_gradient_probe()] if "jax" in backends else []),
    }


def main() -> int:
    probe = run_probe()
    output = REPO_ROOT / "results" / "hazards" / "imaging" / "pixelization" / "probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                name: [asdict(row) if is_dataclass(row) else row for row in rows]
                for name, rows in probe.items()
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
