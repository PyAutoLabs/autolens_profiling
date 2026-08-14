"""Bound PowerLaw omega-series accuracy, prior reachability, and JAX cost.

The NumPy PowerLaw path evaluates the angular factor with SciPy's ``hyp2f1``.
The JAX path uses a recurrence truncated after 20 terms.  This module keeps the
research calculation independent of the finding scanner so the stable finding
ID can outlive changes to the evidence grid.

Run the complete probe from a workspace containing the five sibling PyAuto
repositories::

    python scripts/misc/hazards/power_law_omega.py --ci-report
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import special

FINDING_ID = "component.power-law.series-vs-hyp2f1-divergence"
DEFAULT_ELL_COMPS_SIGMA = 0.3
PUBLIC_SLOPES = (1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 2.99)
FACTORS = (0.0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999)
TERM_COUNTS = (20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240)
ERROR_TOLERANCES = (1.0e-4, 1.0e-6, 1.0e-8)

# This is the cheapest sampled policy that reaches 1e-4 at every bin edge.
# It is a research candidate, not a proposed PyAutoGalaxy default.
POLICY_EDGES = (0.6, 0.7, 0.85, 0.925, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999)
POLICY_TERMS = (20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240)


def profiling_root() -> Path:
    """Return the repository root from this script's location."""

    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


def omega_exact(eiphi: np.ndarray, internal_slope: float, factor: float) -> np.ndarray:
    """Evaluate the PowerLaw angular factor used by the NumPy path."""

    z = np.asarray(eiphi, dtype=complex)
    return z * special.hyp2f1(
        1.0,
        0.5 * internal_slope,
        2.0 - 0.5 * internal_slope,
        -factor * z**2,
    )


def omega_series(
    eiphi: np.ndarray, internal_slope: float, factor: float, n_terms: int
) -> np.ndarray:
    """Evaluate the same finite recurrence used by the JAX implementation."""

    if n_terms < 1:
        raise ValueError("n_terms must be positive")
    z = np.asarray(eiphi, dtype=complex)
    term = z.copy()
    total = z.copy()
    two_minus_slope = 2.0 - internal_slope
    for n in range(1, n_terms):
        two_n = 2.0 * n
        ratio = (two_n - two_minus_slope) / (two_n + two_minus_slope)
        term = -factor * ratio * z**2 * term
        total = total + term
    return total


def relative_l2_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return scale-free complex-vector error."""

    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-30)
    )


def _errors_at_term_counts(
    *, factor: float, public_slope: float, angles: np.ndarray
) -> dict[int, float]:
    """Evaluate all configured truncations in one recurrence pass."""

    internal_slope = public_slope - 1.0
    z = np.exp(1j * angles)
    exact = omega_exact(z, internal_slope, factor)
    term = z.copy()
    total = z.copy()
    errors: dict[int, float] = {}
    requested = set(TERM_COUNTS)
    if 1 in requested:
        errors[1] = relative_l2_error(total, exact)
    two_minus_slope = 2.0 - internal_slope
    for n in range(1, max(TERM_COUNTS)):
        two_n = 2.0 * n
        ratio = (two_n - two_minus_slope) / (two_n + two_minus_slope)
        term = -factor * ratio * z**2 * term
        total = total + term
        evaluated_terms = n + 1
        if evaluated_terms in requested:
            errors[evaluated_terms] = relative_l2_error(total, exact)
    return errors


def convergence_surface() -> list[dict]:
    """Return error versus factor and term count, maximized over the slope prior."""

    angles = np.linspace(-np.pi, np.pi, 129, endpoint=False)
    rows: list[dict] = []
    for factor in FACTORS:
        by_slope = {
            slope: _errors_at_term_counts(
                factor=factor,
                public_slope=slope,
                angles=angles,
            )
            for slope in PUBLIC_SLOPES
        }
        for n_terms in TERM_COUNTS:
            worst_slope = max(PUBLIC_SLOPES, key=lambda slope: by_slope[slope][n_terms])
            rows.append(
                {
                    "factor": factor,
                    "n_terms": n_terms,
                    "max_relative_error": by_slope[worst_slope][n_terms],
                    "worst_public_slope": worst_slope,
                }
            )
    return rows


def minimum_terms(surface: list[dict]) -> list[dict]:
    """Return the first sampled term count meeting each accuracy tolerance."""

    rows: list[dict] = []
    for factor in FACTORS:
        factor_rows = [row for row in surface if row["factor"] == factor]
        for tolerance in ERROR_TOLERANCES:
            passing = [
                row["n_terms"]
                for row in factor_rows
                if row["max_relative_error"] <= tolerance
            ]
            rows.append(
                {
                    "factor": factor,
                    "tolerance": tolerance,
                    "minimum_sampled_terms": min(passing) if passing else None,
                }
            )
    return rows


def _truncated_square_normalization(sigma: float) -> float:
    one_dimensional_mass = math.erf(1.0 / (math.sqrt(2.0) * sigma))
    return one_dimensional_mass**2


def default_prior_annulus_mass(lower: float, upper: float) -> float:
    """Return default ell_comps prior mass in ``lower <= radius < upper <= 1``."""

    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("annulus bounds must satisfy 0 <= lower <= upper <= 1")
    sigma = DEFAULT_ELL_COMPS_SIGMA
    numerator = math.exp(-(lower**2) / (2.0 * sigma**2)) - math.exp(
        -(upper**2) / (2.0 * sigma**2)
    )
    return numerator / _truncated_square_normalization(sigma)


def prior_reachability() -> dict:
    """Integrate 20-term error over the packaged PowerLaw priors."""

    factor_edges = np.linspace(0.0, 1.0, 161)
    slope_edges = np.linspace(1.5, 3.0, 81)
    angles = np.linspace(-np.pi, np.pi, 129, endpoint=False)
    masses = {tolerance: 0.0 for tolerance in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8)}
    for lower, upper in zip(factor_edges[:-1], factor_edges[1:]):
        factor = float(0.5 * (lower + upper))
        radial_mass = default_prior_annulus_mass(float(lower), float(upper))
        for slope_lower, slope_upper in zip(slope_edges[:-1], slope_edges[1:]):
            public_slope = float(0.5 * (slope_lower + slope_upper))
            slope_mass = float((slope_upper - slope_lower) / 1.5)
            z = np.exp(1j * angles)
            exact = omega_exact(z, public_slope - 1.0, factor)
            error = relative_l2_error(
                omega_series(z, public_slope - 1.0, factor, 20), exact
            )
            cell_mass = radial_mass * slope_mass
            for tolerance in masses:
                if error > tolerance:
                    masses[tolerance] += cell_mass

    valid_mass = default_prior_annulus_mass(0.0, 1.0)
    return {
        "prior": {
            "ell_comps_0": "TruncatedGaussian(mean=0, sigma=0.3, limits=[-1, 1])",
            "ell_comps_1": "TruncatedGaussian(mean=0, sigma=0.3, limits=[-1, 1])",
            "public_slope": "Uniform(1.5, 3.0)",
        },
        "quadrature": {
            "factor_bins": len(factor_edges) - 1,
            "slope_bins": len(slope_edges) - 1,
            "angle_count": len(angles),
        },
        "valid_radius_prior_mass": valid_mass,
        "clamped_invalid_radius_prior_mass": 1.0 - valid_mass,
        "absolute_prior_mass_above_error": {
            str(tolerance): mass for tolerance, mass in masses.items()
        },
        "conditional_valid_prior_mass_above_error": {
            str(tolerance): mass / valid_mass for tolerance, mass in masses.items()
        },
        "high_factor_tail_mass": {
            str(lower): default_prior_annulus_mass(lower, 1.0)
            for lower in (0.9, 0.95, 0.99, 0.995, 0.997, 0.999)
        },
    }


def factor_limits_at_twenty_terms(tolerance: float = 1.0e-4) -> list[dict]:
    """Return the largest sampled factor meeting a tolerance at 20 terms."""

    factors = np.linspace(0.0, 0.999, 1000)
    angles = np.linspace(-np.pi, np.pi, 129, endpoint=False)
    rows = []
    for public_slope in PUBLIC_SLOPES:
        internal_slope = public_slope - 1.0
        passing = []
        for factor in factors:
            z = np.exp(1j * angles)
            exact = omega_exact(z, internal_slope, float(factor))
            error = relative_l2_error(
                omega_series(z, internal_slope, float(factor), 20), exact
            )
            if error <= tolerance:
                passing.append(float(factor))
        rows.append(
            {
                "public_slope": public_slope,
                "tolerance": tolerance,
                "largest_sampled_factor": max(passing),
            }
        )
    return rows


def _jax_omega(eiphi, internal_slope, factor, n_terms: int):
    """JAX recurrence with a static scan length."""

    import jax
    import jax.numpy as jnp

    def body(carry, _):
        n, term, total = carry
        two_n = 2 * n
        two_minus_slope = 2.0 - internal_slope
        ratio = (two_n - two_minus_slope) / (two_n + two_minus_slope)
        term = -factor * ratio * eiphi**2 * term
        return (n + 1, term, total + term), None

    (_, _, total), _ = jax.lax.scan(
        body,
        (jnp.asarray(1), eiphi, eiphi),
        xs=None,
        length=n_terms - 1,
    )
    return total


def _jax_policy(eiphi, internal_slope, factor):
    """Research-only fixed-bin candidate retaining static scan lengths."""

    import jax
    import jax.numpy as jnp

    branch_index = jnp.searchsorted(jnp.asarray(POLICY_EDGES), factor, side="left")
    branches = tuple(
        (lambda operands, n_terms=n_terms: _jax_omega(*operands, n_terms))
        for n_terms in POLICY_TERMS
    )
    return jax.lax.switch(branch_index, branches, (eiphi, internal_slope, factor))


def _timed_jax_call(function, *args, warm_repeats: int = 5) -> dict:
    """Return cold and median warm wall time for a JAX callable."""

    import jax

    started = time.perf_counter()
    value = function(*args)
    jax.block_until_ready(value)
    cold_seconds = time.perf_counter() - started
    warm = []
    for _ in range(warm_repeats):
        started = time.perf_counter()
        value = function(*args)
        jax.block_until_ready(value)
        warm.append(time.perf_counter() - started)
    return {
        "cold_seconds": cold_seconds,
        "median_warm_seconds": float(np.median(warm)),
    }


def jax_cost_and_transforms() -> dict:
    """Measure fixed scans and reject candidates that lose grad or vmap."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    angles = jnp.linspace(-jnp.pi, jnp.pi, 129, endpoint=False)
    eiphi = jnp.exp(1j * angles)
    internal_slope = jnp.asarray(1.99)
    factor = jnp.asarray(0.999)
    fixed = {}
    for n_terms in (20, 160, 640, 2560, 10240):
        function = jax.jit(
            lambda z, slope, fac, n_terms=n_terms: _jax_omega(
                z, slope, fac, n_terms
            )
        )
        fixed[str(n_terms)] = _timed_jax_call(
            function, eiphi, internal_slope, factor
        )

    policy_function = jax.jit(_jax_policy)
    policy_timing = _timed_jax_call(
        policy_function, eiphi, internal_slope, factor
    )
    fixed_twenty_warm = fixed["20"]["median_warm_seconds"]
    policy_timing["warm_ratio_to_20_terms"] = (
        policy_timing["median_warm_seconds"] / fixed_twenty_warm
    )

    def objective(fac):
        value = _jax_policy(eiphi, internal_slope, fac)
        return jnp.real(jnp.vdot(value, value))

    transforms = {}
    try:
        transforms["reverse_mode_gradient"] = {
            "supported": True,
            "value_at_0.95": float(jax.grad(objective)(jnp.asarray(0.95))),
        }
    except Exception as exc:  # pragma: no cover - depends on installed JAX
        transforms["reverse_mode_gradient"] = {
            "supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    factors = jnp.asarray((0.6, 0.8, 0.95, 0.999))
    try:
        vmapped = jax.jit(
            jax.vmap(lambda fac: _jax_policy(eiphi, internal_slope, fac))
        )
        transforms["vmap"] = {
            "supported": True,
            **_timed_jax_call(vmapped, factors),
        }
    except Exception as exc:  # pragma: no cover - depends on installed JAX
        transforms["vmap"] = {
            "supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "fixed_scan_timing": fixed,
        "binned_policy": {
            "edges": list(POLICY_EDGES),
            "terms": list(POLICY_TERMS),
            "timing_at_factor_0.999": policy_timing,
            "transforms": transforms,
        },
    }


def complete_likelihood_materiality() -> dict:
    """Compare exact and finite-series paths through a complete FitImaging."""

    import autoarray as aa
    import autolens as al
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    data_native = np.zeros((7, 7))
    data_native[2:5, 2:5] = np.asarray(
        ((0.1, 1.0, 0.0), (0.2, 3.0, 0.3), (0.0, 0.5, 0.1))
    )
    data = aa.Array2D.no_mask(values=data_native, pixel_scales=(1.0, 1.0))
    noise_map = aa.Array2D.full(
        fill_value=2.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
    )
    kernel = aa.Array2D.no_mask(
        values=np.asarray(
            ((0.0, 0.5, 0.0), (0.5, 1.0, 0.5), (0.0, 0.5, 0.0))
        ),
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
    dataset = aa.Imaging(
        data=data,
        psf=aa.Convolver(kernel=kernel),
        noise_map=noise_map,
        over_sample_size_lp=1,
    ).apply_mask(mask=mask)

    def figure_of_merit(factor, public_slope, xp):
        tracer = al.Tracer(
            galaxies=(
                al.Galaxy(
                    redshift=0.5,
                    mass=al.mp.PowerLaw(
                        ell_comps=(factor, 0.0),
                        einstein_radius=1.0,
                        slope=public_slope,
                    ),
                ),
                al.Galaxy(
                    redshift=1.0,
                    bulge=al.lp.SersicSph(
                        centre=(0.13, -0.17),
                        intensity=1.0,
                        effective_radius=0.6,
                        sersic_index=2.0,
                    ),
                ),
            )
        )
        return al.FitImaging(dataset=dataset, tracer=tracer, xp=xp).figure_of_merit

    rows = []
    for public_slope in (2.2, 2.99):
        for factor in (0.6, 0.8, 0.9, 0.95, 0.99, 0.999):
            numpy_value = float(np.asarray(figure_of_merit(factor, public_slope, np)))
            jax_value = float(
                np.asarray(
                    figure_of_merit(
                        jnp.asarray(factor), jnp.asarray(public_slope), jnp
                    )
                )
            )
            rows.append(
                {
                    "factor": factor,
                    "public_slope": public_slope,
                    "numpy_figure_of_merit": numpy_value,
                    "jax_figure_of_merit": jax_value,
                    "absolute_delta_log_likelihood": abs(jax_value - numpy_value),
                }
            )
    return {
        "fixture": "in-memory masked 7x7 Imaging with PowerLaw lens and Sersic source",
        "rows": rows,
        "maximum_absolute_delta_log_likelihood": max(
            row["absolute_delta_log_likelihood"] for row in rows
        ),
    }


def build_report(*, include_runtime: bool) -> dict:
    """Build the reproducible research payload."""

    surface = convergence_surface()
    report = {
        "schema_version": 1,
        "finding_id": FINDING_ID,
        "accuracy": {
            "surface_maximized_over_public_slope_prior": surface,
            "minimum_sampled_terms": minimum_terms(surface),
            "factor_limits_at_20_terms": factor_limits_at_twenty_terms(),
        },
        "prior_reachability": prior_reachability(),
        "candidate_policy": {
            "edges": list(POLICY_EDGES),
            "terms": list(POLICY_TERMS),
            "status": "research_only_pending_runtime_cost",
        },
    }
    if include_runtime:
        report["jax_runtime"] = jax_cost_and_transforms()
        report["complete_likelihood"] = complete_likelihood_materiality()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-report",
        action="store_true",
        help="include JAX and FitImaging probes and print compact tagged JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional output path (default: stdout only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(include_runtime=args.ci_report)
    payload = json.dumps(report, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
    if args.ci_report:
        print(f"POWER_LAW_OMEGA_REPORT={payload}")
    elif args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
