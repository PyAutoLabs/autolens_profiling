"""Audit the JAX MGE Faddeeva routine's region seams and the spherical q clamp.

The NumPy MGE deflection path calls ``scipy.special.wofz`` and takes an exact real
radial branch for circular profiles.  Before PyAutoGalaxy#600 phase B the JAX path
called the hand-rolled rational ``_wofz_rational`` (three ``xp.where``-selected
regions, so its *derivative* jumped at the region boundaries) and evaluated every
spherical profile as an ellipse at the clamp ``q = 0.9999``.  Phase A measured what
those two choices cost under ``jax.grad``/``jax.jacfwd`` and priced a seam-free
replacement (Weideman 1994); phase B landed both, so the "current JAX path" this
module reads out of the library is now ``_wofz_weideman`` taking the exact spherical
branch, and the legs below are the *after* state.  The Weideman implementation kept
here is the independent reference the library routine is checked against, and the
only source of the N=64 order.

As with ``power_law_omega.py`` the research calculation lives here, independent of
the finding scanner, so the stable finding IDs outlive changes to the evidence grid.
``checks/mge_faddeeva.py`` runs the cheap reproducers and owns finding persistence.

Run the complete probe from a workspace containing the sibling PyAuto repositories::

    OMP_NUM_THREADS=1 python scripts/misc/hazards/mge_faddeeva.py --ci-report \
        --output results/hazards/component/mge/faddeeva_audit.json

Without ``--ci-report`` only the dataset-free legs run (seams, the q-sweep of the
elliptical kernel, and the Weideman accuracy study).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import special

SEAM_FINDING_ID = "component.mge.faddeeva-seam-gradient"
CLAMP_FINDING_ID = "component.mge.spherical-clamp-bias"

INV_SQRT_PI = 1.0 / np.sqrt(np.pi)

# Region-boundary constants of the *former* `_wofz_rational` (PyAutoGalaxy
# autogalaxy/profiles/mass/abstract/mge.py, replaced in phase B).  `reg1` selected
# the large-|z| continued fraction, `reg2` region 5, everything else region 6.
# They are kept because they define where the seams *were*: the after-state legs
# sample exactly those loci to show the replacement has no discontinuity there.
R2_EDGES = (2.5, 30.0, 62.0)
Y2_EDGE_MID = 0.072  # inside 2.5 <= r2 < 30
Y2_EDGE_REAL_AXIS = 1.0e-13  # inside 30 <= r2 < 62

SEAM_OFFSETS = (1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6)
SEAM_ANGLES = (0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0, np.pi / 2.0)

# gNFW / gNFWSph fiducials, identical to scripts/lens/deflections/_profiles.py.
KAPPA_S = 0.2
INNER_SLOPE = 1.5
SCALE_RADIUS = 10.0
GAUSSIAN_INTENSITY = 1.0
GAUSSIAN_SIGMA = 1.0

TRANSECT_HALF_WIDTH = 0.05
TRANSECT_STEPS = 2000
FD_STEPS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
CLAMP_Q_LIST = (1.0 - 1.0e-5, 1.0 - 1.0e-6, 1.0 - 1.0e-7, 1.0 - 1.0e-8, 1.0 - 1.0e-9)
WEIDEMAN_ORDERS = (32, 64)
MPMATH_DPS = 40
MPMATH_SAMPLES = 3000
COST_REPEATS = 5


def profiling_root() -> Path:
    """Return the repository root from this script's location."""

    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


def workspace_root() -> Path:
    """Return the workspace holding the sibling PyAuto repositories."""

    root = profiling_root()
    return root.parent if (root.parent / "PyAutoGalaxy").exists() else root.parent


# ---------------------------------------------------------------------------
# Reference Faddeeva quantities
# ---------------------------------------------------------------------------


def wofz_exact(z) -> np.ndarray:
    """``w(z)`` from SciPy (1.3e-14 max relative error against mpmath)."""

    return special.wofz(np.asarray(z, dtype=np.complex128))


def wofz_derivative_exact(z) -> np.ndarray:
    """``w'(z) = -2 z w(z) + 2i/sqrt(pi)``, the identity the seams must satisfy."""

    z = np.asarray(z, dtype=np.complex128)
    return -2.0 * z * wofz_exact(z) + 2.0j * INV_SQRT_PI


def region_labels(z) -> np.ndarray:
    """Region index the former ``_wofz_rational`` chose: 0 = region 6, 1 = region 5, 2 = large-|z|.

    Mirrors the ``reg1``/``reg2`` expressions in the library; the library exposes no
    label of its own, and the predicate is pure arithmetic on ``z``.
    """

    z = np.asarray(z, dtype=np.complex128)
    r2 = np.real(z) ** 2 + np.imag(z) ** 2
    y2 = np.imag(z) ** 2
    reg1 = (r2 >= 62.0) | ((r2 >= 30.0) & (r2 < 62.0) & (y2 >= 1.0e-13))
    reg2 = ((r2 >= 30.0) & (r2 < 62.0) & (y2 < 1.0e-13)) | (
        (r2 >= 2.5) & (r2 < 30.0) & (y2 < 0.072)
    )
    labels = np.zeros(np.shape(z), dtype=np.int8)
    labels[reg2] = 1
    labels[reg1] = 2
    return labels


# ---------------------------------------------------------------------------
# Weideman (1994) rational series -- the replacement candidate
# ---------------------------------------------------------------------------


def weideman_coefficients(n_terms: int) -> tuple[float, np.ndarray]:
    """Return ``(L, a)`` for the Weideman rational approximation of order ``n_terms``.

    Coefficients are the real FFT of a Gaussian sampled on a tangent grid, exactly as
    in Weideman (1994) SIAM J. Numer. Anal. 31, 1497.  They depend on nothing but
    ``n_terms``, so they are computed once with NumPy and reused on every backend.
    """

    m = 2 * n_terms
    m2 = 2 * m
    k = np.arange(-m + 1, m, dtype=np.float64)
    scale = np.sqrt(n_terms / np.sqrt(2.0))
    t = scale * np.tan(k * np.pi / m2)
    f = np.exp(-(t**2)) * (scale**2 + t**2)
    f = np.concatenate((np.zeros(1), f))
    coefficients = np.real(np.fft.fft(np.fft.fftshift(f))) / m2
    return float(scale), np.flipud(coefficients[1 : n_terms + 1]).copy()


_WEIDEMAN_CACHE = {n: weideman_coefficients(n) for n in WEIDEMAN_ORDERS}


def weideman_w(z, xp=np, n_terms: int = 32):
    """``w(z)`` by the Weideman rational series, valid over the upper half-plane.

    A single expression with no region selection, so it is smooth wherever ``w`` is.
    ``zeta_from`` only ever passes ``Im(z) >= 0``, which is where this series holds.
    """

    scale, coefficients = _WEIDEMAN_CACHE[n_terms]
    z = xp.asarray(z, dtype=xp.complex128)
    denominator = scale - 1j * z
    ratio = (scale + 1j * z) / denominator
    polynomial = xp.zeros_like(ratio)
    for coefficient in coefficients:
        polynomial = polynomial * ratio + float(coefficient)
    return 2.0 * polynomial / denominator**2 + INV_SQRT_PI / denominator


# ---------------------------------------------------------------------------
# a. Seam derivative jumps
# ---------------------------------------------------------------------------


def _seam_points() -> list[dict]:
    """Sample pairs straddling every region boundary, with the crossing recorded."""

    points: list[dict] = []
    for r2_edge in R2_EDGES:
        for angle in SEAM_ANGLES:
            for offset in SEAM_OFFSETS:
                below = np.sqrt(r2_edge * (1.0 - offset)) * np.exp(1j * angle)
                above = np.sqrt(r2_edge * (1.0 + offset)) * np.exp(1j * angle)
                points.append(
                    {
                        "seam": f"r2={r2_edge:g}",
                        "angle": float(angle),
                        "offset": offset,
                        "below": complex(below),
                        "above": complex(above),
                    }
                )
    for seam, r2, y2_edge in (
        (f"y2={Y2_EDGE_MID:g} (2.5<=r2<30)", 10.0, Y2_EDGE_MID),
        (f"y2={Y2_EDGE_REAL_AXIS:g} (30<=r2<62)", 45.0, Y2_EDGE_REAL_AXIS),
    ):
        for offset in SEAM_OFFSETS:
            y2_below = y2_edge * (1.0 - offset)
            y2_above = y2_edge * (1.0 + offset)
            below = np.sqrt(r2 - y2_below) + 1j * np.sqrt(y2_below)
            above = np.sqrt(r2 - y2_above) + 1j * np.sqrt(y2_above)
            points.append(
                {
                    "seam": seam,
                    "angle": None,
                    "offset": offset,
                    "below": complex(below),
                    "above": complex(above),
                }
            )
    return points


def _autodiff_derivative(routine, values, jax, jnp) -> np.ndarray:
    """``dw/dx`` by ``jax.jacfwd`` on ``z = x + i y`` with real ``x``, ``y``."""

    def real_imag(x, y):
        w = routine(x + 1j * y, xp=jnp)
        return jnp.stack([jnp.real(w), jnp.imag(w)])

    jacobian = jax.vmap(jax.jacfwd(real_imag, argnums=0))
    values = np.asarray(values, dtype=np.complex128)
    out = np.asarray(
        jacobian(jnp.asarray(np.real(values)), jnp.asarray(np.imag(values))), dtype=np.float64
    )
    return out[:, 0] + 1j * out[:, 1]


def seam_table(routine, *, label: str) -> list[dict]:
    """Value and derivative jumps across every former ``_wofz_rational`` region boundary."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    points = _seam_points()
    below = np.asarray([point["below"] for point in points])
    above = np.asarray([point["above"] for point in points])

    w_below = np.asarray(routine(below, xp=np), dtype=np.complex128)
    w_above = np.asarray(routine(above, xp=np), dtype=np.complex128)
    exact_below = wofz_exact(below)
    exact_above = wofz_exact(above)
    derivative_below = _autodiff_derivative(routine, below, jax, jnp)
    derivative_above = _autodiff_derivative(routine, above, jax, jnp)
    exact_derivative_below = wofz_derivative_exact(below)
    exact_derivative_above = wofz_derivative_exact(above)

    rows: dict[str, dict] = {}
    for index, point in enumerate(points):
        seam = point["seam"]
        row = rows.setdefault(
            seam,
            {
                "seam": seam,
                "routine": label,
                "labels_below": set(),
                "labels_above": set(),
                "max_value_jump": 0.0,
                "max_relative_value_jump": 0.0,
                "max_relative_value_error_below": 0.0,
                "max_relative_value_error_above": 0.0,
                "max_relative_derivative_error_below": 0.0,
                "max_relative_derivative_error_above": 0.0,
                "max_relative_derivative_jump": 0.0,
                "derivative_jump_at_smallest_offset": None,
                "smallest_offset": min(SEAM_OFFSETS),
            },
        )
        row["labels_below"].add(int(region_labels(point["below"])))
        row["labels_above"].add(int(region_labels(point["above"])))

        value_jump = abs(w_above[index] - w_below[index])
        magnitude = max(abs(w_below[index]), 1.0e-300)
        derivative_scale = max(abs(exact_derivative_below[index]), 1.0e-300)
        derivative_jump = abs(derivative_above[index] - derivative_below[index]) / derivative_scale

        row["max_value_jump"] = max(row["max_value_jump"], float(value_jump))
        row["max_relative_value_jump"] = max(
            row["max_relative_value_jump"], float(value_jump / magnitude)
        )
        row["max_relative_value_error_below"] = max(
            row["max_relative_value_error_below"],
            float(abs(w_below[index] - exact_below[index]) / max(abs(exact_below[index]), 1e-300)),
        )
        row["max_relative_value_error_above"] = max(
            row["max_relative_value_error_above"],
            float(abs(w_above[index] - exact_above[index]) / max(abs(exact_above[index]), 1e-300)),
        )
        row["max_relative_derivative_error_below"] = max(
            row["max_relative_derivative_error_below"],
            float(abs(derivative_below[index] - exact_derivative_below[index]) / derivative_scale),
        )
        row["max_relative_derivative_error_above"] = max(
            row["max_relative_derivative_error_above"],
            float(
                abs(derivative_above[index] - exact_derivative_above[index])
                / max(abs(exact_derivative_above[index]), 1.0e-300)
            ),
        )
        row["max_relative_derivative_jump"] = max(
            row["max_relative_derivative_jump"], float(derivative_jump)
        )
        if point["offset"] == min(SEAM_OFFSETS):
            previous = row["derivative_jump_at_smallest_offset"]
            row["derivative_jump_at_smallest_offset"] = (
                float(derivative_jump)
                if previous is None
                else max(previous, float(derivative_jump))
            )

    ordered = []
    for row in rows.values():
        row["labels_below"] = sorted(row["labels_below"])
        row["labels_above"] = sorted(row["labels_above"])
        ordered.append(row)
    return ordered


def seam_error_curve_arrays(routine, *, seam_r2: float = 2.5, angle: float = 0.0):
    """Offsets, exact ``w'`` and autodiff ``w'`` just above one seam, as real 2-vectors.

    The cheap reproducer behind ``component.mge.faddeeva-seam-gradient``: the derivative
    of the routine immediately above a region boundary, where the region it lands in has
    changed but the value has not (to ~1e-6).
    """

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    offsets = np.asarray(SEAM_OFFSETS, dtype=np.float64)
    above = np.sqrt(seam_r2 * (1.0 + offsets)) * np.exp(1j * angle)
    exact = wofz_derivative_exact(above)
    candidate = _autodiff_derivative(routine, above, jax, jnp)
    return (
        offsets,
        np.stack([np.real(exact), np.imag(exact)], axis=1),
        np.stack([np.real(candidate), np.imag(candidate)], axis=1),
    )


def seam_derivative_discontinuity(
    routine, *, seam_r2: float = 2.5, angle: float = 0.0, offset: float = 1.0e-9
) -> dict:
    """How far the routine's ``w'`` jumps across a region boundary, in units of the
    jump ``w'`` genuinely makes there.

    The discriminator phase A settled on: straddle the seam at a relative offset so
    small (1e-9) that the true derivative barely moves, and ask by what factor the
    routine's derivative moves more.  A region-selecting routine steps by the value
    error of the two branches divided by nothing at all -- ``_wofz_rational`` scored
    ~2.2e4 here -- while a single smooth expression scores ~1, because the only
    change is the genuine variation.  The ratio is dimensionless, so it needs no
    magnitude threshold tuned to a particular routine's accuracy.
    """

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    radius = np.sqrt(seam_r2)
    below = radius * np.sqrt(1.0 - offset) * np.exp(1j * angle)
    above = radius * np.sqrt(1.0 + offset) * np.exp(1j * angle)
    pair = np.asarray([below, above])

    exact = wofz_derivative_exact(pair)
    candidate = _autodiff_derivative(routine, pair, jax, jnp)

    scale = max(float(np.abs(exact[0])), 1.0e-300)
    true_jump = float(np.abs(exact[1] - exact[0])) / scale
    routine_jump = float(np.abs(candidate[1] - candidate[0])) / scale
    return {
        "seam_r2": seam_r2,
        "angle": angle,
        "offset": offset,
        "true_relative_jump": true_jump,
        "routine_relative_jump": routine_jump,
        "excess_factor": routine_jump / max(true_jump, 1.0e-300),
    }


# ---------------------------------------------------------------------------
# e. Weideman accuracy and cost
# ---------------------------------------------------------------------------


def mpmath_reference(values: np.ndarray) -> np.ndarray:
    """``w(z)`` at ``mpmath`` dps 40, the accuracy reference for every routine."""

    import mpmath

    with mpmath.workdps(MPMATH_DPS):
        out = np.empty(values.shape, dtype=np.complex128)
        for index, value in enumerate(values.reshape(-1)):
            z = mpmath.mpc(float(np.real(value)), float(np.imag(value)))
            out.reshape(-1)[index] = complex(mpmath.exp(-(z**2)) * mpmath.erfc(-1j * z))
    return out


def logspaced_domain(count: int = 600) -> np.ndarray:
    """A log-spaced ``|z|`` sweep at several arguments in the upper half-plane."""

    radii = np.logspace(-6.0, 2.5, count)
    angles = (0.0, 1.0e-7, 0.01, np.pi / 8.0, np.pi / 4.0, np.pi / 2.0)
    return np.concatenate([radii * np.exp(1j * angle) for angle in angles])


def accuracy_rows(values: np.ndarray, *, domain: str) -> list[dict]:
    """Relative error of each routine against mpmath over one input domain."""

    from autogalaxy.profiles.mass.abstract.mge import _wofz_weideman

    reference = mpmath_reference(values)
    magnitude = np.maximum(np.abs(reference), 1.0e-300)
    routines = {
        "scipy.special.wofz": lambda z: wofz_exact(z),
        "_wofz_weideman (current JAX path)": lambda z: np.asarray(_wofz_weideman(z, xp=np)),
    }
    for n_terms in WEIDEMAN_ORDERS:
        routines[f"weideman_{n_terms}"] = lambda z, n_terms=n_terms: np.asarray(
            weideman_w(z, xp=np, n_terms=n_terms)
        )

    rows = []
    for name, routine in routines.items():
        error = np.abs(routine(values) - reference) / magnitude
        worst = int(np.argmax(error))
        rows.append(
            {
                "domain": domain,
                "routine": name,
                "sample_count": int(values.size),
                "max_relative_error": float(np.max(error)),
                "median_relative_error": float(np.median(error)),
                "worst_z": [float(np.real(values[worst])), float(np.imag(values[worst]))],
                "worst_abs_z": float(np.abs(values[worst])),
            }
        )
    return rows


def _timed(function, *args, repeats: int = COST_REPEATS) -> dict:
    """Cold (compile included) and median warm wall time."""

    import jax

    started = time.perf_counter()
    value = function(*args)
    jax.block_until_ready(value)
    cold = time.perf_counter() - started
    warm = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function(*args)
        jax.block_until_ready(value)
        warm.append(time.perf_counter() - started)
    return {"cold_seconds": cold, "median_warm_seconds": float(np.median(warm))}


def cost_rows(block: np.ndarray) -> list[dict]:
    """JAX jit compile + steady-state cost on the real ``(30, N)`` MGE block."""

    import jax
    import jax.numpy as jnp
    from autogalaxy.profiles.mass.abstract.mge import _wofz_weideman

    jax.config.update("jax_enable_x64", True)
    block_jax = jnp.asarray(block)
    rows = []

    timing = _timed(jax.jit(lambda z: _wofz_weideman(z, xp=jnp)), block_jax)
    rows.append({"routine": "_wofz_weideman", "backend": "jax", **timing})
    for n_terms in WEIDEMAN_ORDERS:
        timing = _timed(
            jax.jit(lambda z, n_terms=n_terms: weideman_w(z, xp=jnp, n_terms=n_terms)), block_jax
        )
        rows.append({"routine": f"weideman_{n_terms}", "backend": "jax", **timing})

    for name, routine in (
        ("scipy.special.wofz", wofz_exact),
        ("_wofz_weideman", lambda z: _wofz_weideman(z, xp=np)),
    ) + tuple(
        (f"weideman_{n}", lambda z, n=n: weideman_w(z, xp=np, n_terms=n)) for n in WEIDEMAN_ORDERS
    ):
        routine(block)
        samples = []
        for _ in range(COST_REPEATS):
            started = time.perf_counter()
            routine(block)
            samples.append(time.perf_counter() - started)
        rows.append(
            {
                "routine": name,
                "backend": "numpy",
                "cold_seconds": None,
                "median_warm_seconds": float(np.median(samples)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Dataset + profile fixtures
# ---------------------------------------------------------------------------


def load_deflection_driver():
    """Import ``scripts/lens/deflections/_driver.py`` so the grid is not transcribed."""

    import importlib.util
    import sys

    root = profiling_root()
    for path in (root, root / "scripts" / "misc", root / "scripts" / "lens" / "deflections"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    driver_path = root / "scripts" / "lens" / "deflections" / "_driver.py"
    spec = importlib.util.spec_from_file_location("_deflection_driver", driver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hst_grid_and_samples():
    """Return the hst ray-trace ``Grid2D`` and the driver's 16 pin coordinates."""

    import os

    driver = load_deflection_driver()
    previous = os.getcwd()
    os.chdir(profiling_root())
    try:
        dataset, _ = driver.build_dataset("hst", workspace_root())
    finally:
        os.chdir(previous)
    return dataset.grids.pixelization, np.asarray(driver.PIN_COORDINATES, dtype=np.float64)


def gnfw_profile(centre=(0.0, 0.0), ell_comps=None):
    """The ``_profiles.py`` gNFW fiducial (MGE-30), optionally re-centred."""

    import autolens as al

    if ell_comps is None:
        ell_comps = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)
    return al.mp.gNFW(
        centre=centre,
        ell_comps=(float(ell_comps[0]), float(ell_comps[1])),
        kappa_s=KAPPA_S,
        inner_slope=INNER_SLOPE,
        scale_radius=SCALE_RADIUS,
    )


def _irregular(values, xp=np):
    import autoarray as aa

    return aa.Grid2DIrregular(values=xp.asarray(np.asarray(values, dtype=np.float64)))


# ---------------------------------------------------------------------------
# d. Spherical clamp
# ---------------------------------------------------------------------------


def _cross_axis(coordinates: np.ndarray, deflections: np.ndarray) -> np.ndarray:
    """Deflection component perpendicular to the radius; exactly zero when q = 1."""

    y, x = coordinates[:, 0], coordinates[:, 1]
    radius = np.maximum(np.hypot(y, x), 1.0e-300)
    return (deflections[:, 0] * x - deflections[:, 1] * y) / radius


def clamp_bias_rows(sample_coordinates: np.ndarray, grid) -> dict:
    """JAX clamped-elliptical versus the exact spherical form, at q = 1 profiles."""

    import autolens as al
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    profiles = {
        "gNFWSph (MGE-30)": al.mp.gNFWSph(
            centre=(0.0, 0.0),
            kappa_s=KAPPA_S,
            inner_slope=INNER_SLOPE,
            scale_radius=SCALE_RADIUS,
        ),
        "Gaussian (ell_comps=(0,0))": al.mp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            intensity=GAUSSIAN_INTENSITY,
            sigma=GAUSSIAN_SIGMA,
        ),
    }

    point_rows = []
    field_rows = []
    for name, profile in profiles.items():
        exact = np.asarray(
            profile.deflections_yx_2d_from(grid=_irregular(sample_coordinates), xp=np).array
        )
        clamped = np.asarray(
            profile.deflections_yx_2d_from(
                grid=_irregular(sample_coordinates, xp=jnp), xp=jnp
            ).array
        )
        magnitude = np.maximum(np.linalg.norm(exact, axis=1), 1.0e-300)
        relative = np.linalg.norm(clamped - exact, axis=1) / magnitude
        cross = _cross_axis(sample_coordinates, clamped)
        for index, coordinate in enumerate(sample_coordinates):
            point_rows.append(
                {
                    "profile": name,
                    "y": float(coordinate[0]),
                    "x": float(coordinate[1]),
                    "radius": float(np.hypot(*coordinate)),
                    "exact_magnitude": float(magnitude[index]),
                    "relative_bias": float(relative[index]),
                    "cross_axis_deflection": float(cross[index]),
                }
            )

        exact_field = np.asarray(profile.deflections_yx_2d_from(grid=grid, xp=np).array)
        grid_values = np.asarray(grid.array)
        clamped_field = np.asarray(
            profile.deflections_yx_2d_from(grid=_irregular(grid_values, xp=jnp), xp=jnp).array
        )
        field_magnitude = np.maximum(np.linalg.norm(exact_field, axis=1), 1.0e-300)
        field_relative = np.linalg.norm(clamped_field - exact_field, axis=1) / field_magnitude
        field_rows.append(
            {
                "profile": name,
                "grid": "hst pixelization Grid2D",
                "point_count": int(grid_values.shape[0]),
                "max_relative_bias": float(np.max(field_relative)),
                "median_relative_bias": float(np.median(field_relative)),
                "max_abs_cross_axis": float(
                    np.max(np.abs(_cross_axis(grid_values, clamped_field)))
                ),
            }
        )
    return {"sample_points": point_rows, "hst_field": field_rows}


def clamp_q_sweep(sample_coordinates: np.ndarray) -> list[dict]:
    """Elliptical kernel accuracy as q -> 1, with the library clamp bypassed.

    The clamp is bypassed by a probe-local ``MGEDecomposer`` subclass injected into
    ``gnfw.py``'s module namespace for the duration of one call -- the library source
    is never edited, and the injected decomposer differs from the real one only in the
    axis ratio it reports.
    """

    import autolens as al
    import jax
    import jax.numpy as jnp
    from autogalaxy.profiles.mass import MGEDecomposer
    from autogalaxy.profiles.mass.dark import gnfw as gnfw_module

    jax.config.update("jax_enable_x64", True)

    class FixedAxisRatioDecomposer(MGEDecomposer):
        """``MGEDecomposer`` reporting a caller-chosen axis ratio instead of the clamp."""

        axis_ratio_value = 0.9999

        def axis_ratio(self, xp=np):
            return xp.asarray(self.axis_ratio_value, dtype=xp.float64)

    profile = al.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=KAPPA_S, inner_slope=INNER_SLOPE, scale_radius=SCALE_RADIUS
    )
    exact = np.asarray(
        profile.deflections_yx_2d_from(grid=_irregular(sample_coordinates), xp=np).array
    )
    magnitude = np.maximum(np.linalg.norm(exact, axis=1), 1.0e-300)

    rows = []
    original = gnfw_module.MGEDecomposer
    try:
        for axis_ratio in (0.9999,) + CLAMP_Q_LIST:

            def factory(mass_profile, axis_ratio=axis_ratio):
                decomposer = FixedAxisRatioDecomposer(mass_profile=mass_profile)
                decomposer.axis_ratio_value = axis_ratio
                return decomposer

            gnfw_module.MGEDecomposer = factory
            values = np.asarray(
                profile.deflections_yx_2d_from(
                    grid=_irregular(sample_coordinates, xp=jnp), xp=jnp
                ).array
            )
            relative = np.linalg.norm(values - exact, axis=1) / magnitude
            rows.append(
                {
                    "axis_ratio": float(axis_ratio),
                    "one_minus_q": float(1.0 - axis_ratio),
                    "max_relative_error": float(np.max(relative)),
                    "median_relative_error": float(np.median(relative)),
                    "max_abs_cross_axis": float(
                        np.max(np.abs(_cross_axis(sample_coordinates, values)))
                    ),
                    "n_non_finite": int(np.sum(~np.isfinite(values))),
                }
            )
    finally:
        gnfw_module.MGEDecomposer = original
    return rows


def clamp_q_sweep_gaussian(sample_coordinates: np.ndarray) -> dict:
    """The same q -> 1 sweep for ``Gaussian``, whose clamp lives on the profile itself."""

    import autolens as al
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    try:

        class FixedAxisRatioGaussian(al.mp.Gaussian):
            axis_ratio_value = 0.9999

            def axis_ratio(self, xp=np):
                return xp.asarray(self.axis_ratio_value, dtype=xp.float64)

        reference = al.mp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            intensity=GAUSSIAN_INTENSITY,
            sigma=GAUSSIAN_SIGMA,
        )
        exact = np.asarray(
            reference.deflections_yx_2d_from(grid=_irregular(sample_coordinates), xp=np).array
        )
        magnitude = np.maximum(np.linalg.norm(exact, axis=1), 1.0e-300)
        rows = []
        for axis_ratio in (0.9999,) + CLAMP_Q_LIST:
            profile = FixedAxisRatioGaussian(
                centre=(0.0, 0.0),
                ell_comps=(0.0, 0.0),
                intensity=GAUSSIAN_INTENSITY,
                sigma=GAUSSIAN_SIGMA,
            )
            profile.axis_ratio_value = axis_ratio
            values = np.asarray(
                profile.deflections_yx_2d_from(
                    grid=_irregular(sample_coordinates, xp=jnp), xp=jnp
                ).array
            )
            relative = np.linalg.norm(values - exact, axis=1) / magnitude
            rows.append(
                {
                    "axis_ratio": float(axis_ratio),
                    "one_minus_q": float(1.0 - axis_ratio),
                    "max_relative_error": float(np.max(relative)),
                    "median_relative_error": float(np.median(relative)),
                    "n_non_finite": int(np.sum(~np.isfinite(values))),
                }
            )
        return {"measured": True, "rows": rows}
    except Exception as exc:  # pragma: no cover - depends on the installed profile API
        return {"measured": False, "reason": f"{type(exc).__name__}: {exc}", "rows": []}


# ---------------------------------------------------------------------------
# b. Deflection-gradient impact
# ---------------------------------------------------------------------------

PARAMETER_NAMES = ("centre_y", "centre_x", "ell_comps_0", "ell_comps_1", "scale_radius")


def _parameter_vector() -> np.ndarray:
    import autolens as al

    ell_comps = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)
    return np.asarray(
        [0.0, 0.0, float(ell_comps[0]), float(ell_comps[1]), SCALE_RADIUS], dtype=np.float64
    )


def _deflections_from_parameters(parameters, grid, xp):
    import autolens as al

    profile = al.mp.gNFW(
        centre=(parameters[0], parameters[1]),
        ell_comps=(parameters[2], parameters[3]),
        kappa_s=KAPPA_S,
        inner_slope=INNER_SLOPE,
        scale_radius=parameters[4],
    )
    return profile.deflections_yx_2d_from(grid=grid, xp=xp).array


def _finite_difference_jacobian(parameters, grid, xp, step: float) -> np.ndarray:
    """Central-difference Jacobian, shape ``(n_points, 2, n_parameters)``."""

    columns = []
    for index in range(parameters.size):
        forward = parameters.copy()
        backward = parameters.copy()
        forward[index] += step
        backward[index] -= step
        plus = np.asarray(_deflections_from_parameters(xp.asarray(forward), grid, xp))
        minus = np.asarray(_deflections_from_parameters(xp.asarray(backward), grid, xp))
        columns.append((plus - minus) / (2.0 * step))
    return np.stack(columns, axis=-1)


def deflection_jacobian_rows(grid) -> dict:
    """``jax.jacfwd`` against finite differences on both backends, over a step sweep."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    coordinates = np.asarray(grid.array)
    grid_numpy = _irregular(coordinates)
    grid_jax = _irregular(coordinates, xp=jnp)
    parameters = _parameter_vector()

    autodiff = np.asarray(
        jax.jacfwd(lambda p: _deflections_from_parameters(p, grid_jax, jnp))(
            jnp.asarray(parameters)
        )
    )
    rows = []
    for backend_name, backend_grid, backend in (
        ("numpy(scipy) FD", grid_numpy, np),
        ("jax FD", grid_jax, jnp),
    ):
        for step in FD_STEPS:
            finite = _finite_difference_jacobian(parameters, backend_grid, backend, step)
            for index, name in enumerate(PARAMETER_NAMES):
                reference = finite[:, :, index]
                candidate = autodiff[:, :, index]
                scale = max(float(np.linalg.norm(reference)), 1.0e-300)
                difference = np.abs(candidate - reference)
                point_scale = max(float(np.max(np.abs(reference))), 1.0e-300)
                rows.append(
                    {
                        "comparison": f"jax jacfwd vs {backend_name}",
                        "parameter": name,
                        "step": step,
                        "relative_l2_difference": float(
                            np.linalg.norm(candidate - reference) / scale
                        ),
                        "max_abs_difference": float(np.max(difference)),
                        "max_relative_point_difference": float(np.max(difference) / point_scale),
                        "n_points_above_1_percent": int(
                            np.sum(np.max(difference, axis=1) > 0.01 * point_scale)
                        ),
                    }
                )
    return {
        "parameters": list(PARAMETER_NAMES),
        "parameter_values": parameters.tolist(),
        "n_non_finite_in_autodiff": int(np.sum(~np.isfinite(autodiff))),
        "rows": rows,
    }


def _captured_region_labels(profile, grid_numpy) -> tuple[np.ndarray, np.ndarray]:
    """Deflections on the numpy path plus the region labels of every ``w`` argument.

    The ``z`` arrays are captured by wrapping the library's own ``_wofz`` /
    ``_wofz_masked`` for the duration of one call, so no part of ``zeta_from`` is
    transcribed here; only the full ``(n_gaussians, n_pixels)`` blocks are kept.
    """

    from autogalaxy.profiles.mass.abstract import mge as mge_module

    captured: list[np.ndarray] = []
    original_wofz = mge_module._wofz
    original_masked = mge_module._wofz_masked

    def wofz_spy(z, xp=np):
        captured.append(np.asarray(z, dtype=np.complex128))
        return original_wofz(z, xp=xp)

    def masked_spy(z, exp_term, threshold: float = 1.0e-18):
        captured.append(np.asarray(z, dtype=np.complex128))
        return original_masked(z, exp_term, threshold)

    mge_module._wofz = wofz_spy
    mge_module._wofz_masked = masked_spy
    try:
        deflections = np.asarray(
            profile.deflections_yx_2d_from(grid=grid_numpy, xp=np).array, dtype=np.float64
        )
    finally:
        mge_module._wofz = original_wofz
        mge_module._wofz_masked = original_masked

    n_points = np.asarray(grid_numpy.array).shape[0]
    blocks = [block for block in captured if block.ndim == 2 and block.shape[1] == n_points]
    labels = np.concatenate([region_labels(block).reshape(-1) for block in blocks])
    return deflections, labels


def centre_transect(grid, steps: int = TRANSECT_STEPS) -> dict:
    """Region-label churn and AD-gradient continuity along a ``centre_x`` transect."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    coordinates = np.asarray(grid.array)
    grid_numpy = _irregular(coordinates)
    grid_jax = _irregular(coordinates, xp=jnp)
    centres = np.linspace(-TRANSECT_HALF_WIDTH, TRANSECT_HALF_WIDTH, steps)

    def objective(centre_x):
        parameters = jnp.asarray(_parameter_vector()).at[1].set(centre_x)
        deflections = _deflections_from_parameters(parameters, grid_jax, jnp)
        return jnp.sum(deflections**2)

    gradient = jax.jit(jax.grad(objective))

    objective_numpy = np.empty(steps)
    autodiff_gradient = np.empty(steps)
    label_change_fraction = np.empty(steps)
    label_change_fraction[0] = np.nan
    previous_labels = None
    started = time.perf_counter()
    for index, centre_x in enumerate(centres):
        profile = gnfw_profile(centre=(0.0, float(centre_x)))
        deflections, labels = _captured_region_labels(profile, grid_numpy)
        objective_numpy[index] = float(np.sum(deflections**2))
        if previous_labels is not None:
            label_change_fraction[index] = float(np.mean(labels != previous_labels))
        previous_labels = labels
        autodiff_gradient[index] = float(gradient(jnp.asarray(float(centre_x))))

    spacing = float(centres[1] - centres[0])
    finite_difference = np.full(steps, np.nan)
    finite_difference[1:-1] = (objective_numpy[2:] - objective_numpy[:-2]) / (2.0 * spacing)

    interior = slice(1, -1)
    scale = float(np.max(np.abs(finite_difference[interior])))
    residual = np.abs(autodiff_gradient[interior] - finite_difference[interior]) / max(
        scale, 1e-300
    )
    step_jump = np.abs(np.diff(autodiff_gradient)) / max(scale, 1e-300)
    smooth_jump = np.abs(np.diff(finite_difference[interior])) / max(scale, 1e-300)
    changed = label_change_fraction[1:]
    non_finite = ~np.isfinite(autodiff_gradient)
    finite_jump = np.isfinite(step_jump)
    with np.errstate(invalid="ignore"):
        jump_where_labels_change = step_jump[finite_jump & (np.nan_to_num(changed) > 0.0)]
        jump_where_labels_static = step_jump[finite_jump & (np.nan_to_num(changed) == 0.0)]

    return {
        "centre_x_range": [-TRANSECT_HALF_WIDTH, TRANSECT_HALF_WIDTH],
        "steps": steps,
        "step_size_arcsec": spacing,
        "wall_seconds": time.perf_counter() - started,
        "objective": "sum of squared (y,x) deflections over the hst Grid2D",
        "gradient_scale": scale,
        "max_label_change_fraction": float(np.nanmax(label_change_fraction)),
        "median_label_change_fraction": float(np.nanmedian(label_change_fraction)),
        "steps_with_label_change": int(np.nansum(label_change_fraction > 0.0)),
        "max_relative_ad_vs_fd_residual": float(np.nanmax(residual)),
        "median_relative_ad_vs_fd_residual": float(np.nanmedian(residual)),
        "max_relative_step_jump_autodiff": float(np.nanmax(step_jump)),
        "median_relative_step_jump_autodiff": float(np.nanmedian(step_jump)),
        "max_relative_step_jump_finite_difference": float(np.nanmax(smooth_jump)),
        "median_relative_step_jump_finite_difference": float(np.nanmedian(smooth_jump)),
        "max_relative_step_jump_where_labels_change": (
            float(np.max(jump_where_labels_change)) if jump_where_labels_change.size else None
        ),
        "median_relative_step_jump_where_labels_change": (
            float(np.median(jump_where_labels_change)) if jump_where_labels_change.size else None
        ),
        "median_relative_step_jump_where_labels_static": (
            float(np.median(jump_where_labels_static)) if jump_where_labels_static.size else None
        ),
        "n_non_finite_autodiff_gradient": int(np.sum(non_finite)),
        "non_finite_autodiff_centre_x": centres[non_finite].tolist(),
        "series": {
            "centre_x": centres.tolist(),
            "autodiff_gradient": autodiff_gradient.tolist(),
            "finite_difference_gradient": [
                None if not np.isfinite(value) else float(value) for value in finite_difference
            ],
            "label_change_fraction": [
                None if not np.isfinite(value) else float(value) for value in label_change_fraction
            ],
        },
    }


# ---------------------------------------------------------------------------
# c. Likelihood-level transect (bounded)
# ---------------------------------------------------------------------------

LIKELIHOOD_SHAPE = (21, 21)
LIKELIHOOD_PIXEL_SCALE = 0.3
LIKELIHOOD_MASK_RADIUS = 2.4
LIKELIHOOD_STEPS = 400


def _likelihood_fixture():
    """A small in-memory masked ``Imaging`` fixture, deterministic and self-contained."""

    import autoarray as aa

    rng = np.random.default_rng(107)
    data_native = rng.normal(loc=1.0, scale=0.1, size=LIKELIHOOD_SHAPE)
    data = aa.Array2D.no_mask(
        values=data_native, pixel_scales=(LIKELIHOOD_PIXEL_SCALE, LIKELIHOOD_PIXEL_SCALE)
    )
    noise_map = aa.Array2D.full(
        fill_value=0.1,
        shape_native=LIKELIHOOD_SHAPE,
        pixel_scales=(LIKELIHOOD_PIXEL_SCALE, LIKELIHOOD_PIXEL_SCALE),
    )
    kernel = aa.Array2D.no_mask(
        values=np.asarray(((0.0, 0.5, 0.0), (0.5, 1.0, 0.5), (0.0, 0.5, 0.0))),
        pixel_scales=(LIKELIHOOD_PIXEL_SCALE, LIKELIHOOD_PIXEL_SCALE),
    )
    mask = aa.Mask2D.circular(
        shape_native=LIKELIHOOD_SHAPE,
        pixel_scales=(LIKELIHOOD_PIXEL_SCALE, LIKELIHOOD_PIXEL_SCALE),
        radius=LIKELIHOOD_MASK_RADIUS,
    )
    dataset = aa.Imaging(
        data=data,
        psf=aa.Convolver(kernel=kernel),
        noise_map=noise_map,
        over_sample_size_lp=1,
    ).apply_mask(mask=mask)
    return dataset


def likelihood_transect(steps: int = LIKELIHOOD_STEPS) -> dict:
    """``jax.grad`` of a complete gNFW-lens log-likelihood along the same transect."""

    started = time.perf_counter()
    try:
        import autolens as al
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dataset = _likelihood_fixture()

        def figure_of_merit(centre_x, xp):
            tracer = al.Tracer(
                galaxies=(
                    al.Galaxy(
                        redshift=0.5,
                        mass=al.mp.gNFW(
                            centre=(0.0, centre_x),
                            ell_comps=(0.15, 0.05),
                            kappa_s=KAPPA_S,
                            inner_slope=INNER_SLOPE,
                            scale_radius=SCALE_RADIUS,
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

        gradient = jax.jit(jax.grad(lambda centre_x: figure_of_merit(centre_x, jnp)))
        centres = np.linspace(-TRANSECT_HALF_WIDTH, TRANSECT_HALF_WIDTH, steps)
        values = np.empty(steps)
        gradients = np.empty(steps)
        for index, centre_x in enumerate(centres):
            values[index] = float(np.asarray(figure_of_merit(float(centre_x), np)))
            gradients[index] = float(gradient(jnp.asarray(float(centre_x))))

        spacing = float(centres[1] - centres[0])
        finite_difference = np.full(steps, np.nan)
        finite_difference[1:-1] = (values[2:] - values[:-2]) / (2.0 * spacing)
        scale = float(np.max(np.abs(finite_difference[1:-1])))
        step_jump = np.abs(np.diff(gradients)) / max(scale, 1.0e-300)
        smooth_jump = np.abs(np.diff(finite_difference[1:-1])) / max(scale, 1.0e-300)
        threshold = 5.0 * float(np.median(smooth_jump))
        residual = np.abs(gradients[1:-1] - finite_difference[1:-1]) / max(scale, 1.0e-300)
        return {
            "measured": True,
            "fixture": (
                f"in-memory {LIKELIHOOD_SHAPE[0]}x{LIKELIHOOD_SHAPE[1]} masked Imaging "
                f'(radius {LIKELIHOOD_MASK_RADIUS}", pixel scale {LIKELIHOOD_PIXEL_SCALE}"), '
                "gNFW lens + SersicSph source, no inversion"
            ),
            "steps": steps,
            "wall_seconds": time.perf_counter() - started,
            "log_likelihood_gradient_scale": scale,
            "max_relative_step_jump_autodiff": float(np.max(step_jump)),
            "median_relative_step_jump_autodiff": float(np.median(step_jump)),
            "max_relative_step_jump_finite_difference": float(np.max(smooth_jump)),
            "median_relative_step_jump_finite_difference": float(np.median(smooth_jump)),
            "max_relative_ad_vs_fd_residual": float(np.nanmax(residual)),
            "median_relative_ad_vs_fd_residual": float(np.nanmedian(residual)),
            "kink_threshold_relative": threshold,
            "kink_count": int(np.sum(step_jump > threshold)),
            "kink_count_finite_difference_baseline": int(np.sum(smooth_jump > threshold)),
            "n_non_finite_gradient": int(np.sum(~np.isfinite(gradients))),
            "series": {
                "centre_x": centres.tolist(),
                "autodiff_gradient": gradients.tolist(),
                "figure_of_merit": values.tolist(),
            },
        }
    except Exception as exc:  # pragma: no cover - depends on the installed likelihood API
        return {
            "measured": False,
            "reason": f"not measured -- {type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
        }


# ---------------------------------------------------------------------------
# The phase-B premise: is `ell_comps` static under `jax.vmap` for a *Sph class?
# ---------------------------------------------------------------------------


def vmap_ell_comps_staticness() -> dict:
    """Show ``gNFWSph.ell_comps`` stays a Python float tuple under ``jax.vmap``."""

    import autolens as al
    import jax
    import jax.numpy as jnp
    from autogalaxy.profiles.mass.abstract.mge import _is_circular

    jax.config.update("jax_enable_x64", True)
    observed: dict[str, object] = {}

    def build(kappa_s, scale_radius):
        profile = al.mp.gNFWSph(
            centre=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=INNER_SLOPE,
            scale_radius=scale_radius,
        )
        observed["ell_comps_component_type"] = type(profile.ell_comps[0]).__name__
        observed["ell_comps_is_python_float"] = type(profile.ell_comps[0]) is float
        observed["ell_comps_value"] = [float(profile.ell_comps[0]), float(profile.ell_comps[1])]
        observed["kappa_s_type_under_vmap"] = type(profile.kappa_s).__name__
        circular = _is_circular(profile.ell_comps)
        observed["is_circular"] = bool(circular)
        observed["is_circular_type"] = type(circular).__name__
        return profile.kappa_s * profile.scale_radius

    result = jax.vmap(build)(jnp.asarray([0.1, 0.2]), jnp.asarray([5.0, 10.0]))
    observed["vmap_output"] = np.asarray(result).tolist()

    model_probe: dict[str, object]
    try:
        import autofit as af
        from autofit.jax import register_model

        model = af.Model(
            al.mp.gNFWSph,
            centre=(0.0, 0.0),
            kappa_s=af.UniformPrior(lower_limit=0.05, upper_limit=0.5),
            inner_slope=INNER_SLOPE,
            scale_radius=af.UniformPrior(lower_limit=1.0, upper_limit=20.0),
        )
        register_model(model)
        instance = model.instance_from_prior_medians()
        tree = jax.tree_util.tree_map(jnp.asarray, instance)
        model_probe = {
            "available": True,
            "ell_comps_component_type": type(tree.ell_comps[0]).__name__,
            "ell_comps_is_python_float": type(tree.ell_comps[0]) is float,
            "kappa_s_type": type(tree.kappa_s).__name__,
        }
    except Exception as exc:  # pragma: no cover - depends on the installed autofit
        model_probe = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    return {"direct_construction_under_vmap": observed, "autofit_model_pytree": model_probe}


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def mge_domain_block(grid) -> np.ndarray:
    """The real ``(n_gaussians, n_pixels)`` ``w`` arguments of a gNFW MGE-30 hst call."""

    from autogalaxy.profiles.mass.abstract import mge as mge_module

    captured: list[np.ndarray] = []
    original_wofz = mge_module._wofz
    original_masked = mge_module._wofz_masked

    def wofz_spy(z, xp=np):
        captured.append(np.asarray(z, dtype=np.complex128))
        return original_wofz(z, xp=xp)

    def masked_spy(z, exp_term, threshold: float = 1.0e-18):
        captured.append(np.asarray(z, dtype=np.complex128))
        return original_masked(z, exp_term, threshold)

    mge_module._wofz = wofz_spy
    mge_module._wofz_masked = masked_spy
    try:
        gnfw_profile().deflections_yx_2d_from(grid=grid, xp=np)
    finally:
        mge_module._wofz = original_wofz
        mge_module._wofz_masked = original_masked

    n_points = np.asarray(grid.array).shape[0]
    blocks = [block for block in captured if block.ndim == 2 and block.shape[1] == n_points]
    return blocks[0]


def mge_domain_samples(grid, count: int = MPMATH_SAMPLES) -> tuple[np.ndarray, dict]:
    """A stratified subsample of the real ``zeta_from`` inputs, for the mpmath study."""

    from autogalaxy.profiles.mass.abstract import mge as mge_module

    captured: list[np.ndarray] = []
    original_wofz = mge_module._wofz
    original_masked = mge_module._wofz_masked

    def wofz_spy(z, xp=np):
        captured.append(np.asarray(z, dtype=np.complex128))
        return original_wofz(z, xp=xp)

    def masked_spy(z, exp_term, threshold: float = 1.0e-18):
        captured.append(np.asarray(z, dtype=np.complex128))
        return original_masked(z, exp_term, threshold)

    mge_module._wofz = wofz_spy
    mge_module._wofz_masked = masked_spy
    try:
        gnfw_profile().deflections_yx_2d_from(grid=grid, xp=np)
    finally:
        mge_module._wofz = original_wofz
        mge_module._wofz_masked = original_masked

    n_points = np.asarray(grid.array).shape[0]
    blocks = [block for block in captured if block.ndim == 2 and block.shape[1] == n_points]
    values = np.concatenate([block.reshape(-1) for block in blocks])
    labels = region_labels(values)
    rng = np.random.default_rng(107)
    selection = rng.choice(values.size, size=min(count, values.size), replace=False)
    sample = values[selection]
    return sample, {
        "block_shapes": [list(block.shape) for block in blocks],
        "total_values": int(values.size),
        "sampled_values": int(sample.size),
        "abs_z_min": float(np.min(np.abs(values))),
        "abs_z_max": float(np.max(np.abs(values))),
        "region_fractions": {
            "region_6": float(np.mean(labels == 0)),
            "region_5": float(np.mean(labels == 1)),
            "large_z": float(np.mean(labels == 2)),
        },
    }


def build_report(*, include_runtime: bool) -> dict:
    """Build the reproducible research payload."""

    from autogalaxy.profiles.mass.abstract.mge import _wofz_weideman

    report = {
        "schema_version": 1,
        "finding_ids": [SEAM_FINDING_ID, CLAMP_FINDING_ID],
        "seams": {
            "boundaries": {
                "r2": list(R2_EDGES),
                "y2_mid": Y2_EDGE_MID,
                "y2_real_axis": Y2_EDGE_REAL_AXIS,
            },
            "offsets": list(SEAM_OFFSETS),
            "current_routine": seam_table(_wofz_weideman, label="_wofz_weideman"),
            "weideman_32": seam_table(
                lambda z, xp=np: weideman_w(z, xp=xp, n_terms=32), label="weideman_32"
            ),
            "weideman_64": seam_table(
                lambda z, xp=np: weideman_w(z, xp=xp, n_terms=64), label="weideman_64"
            ),
        },
        "accuracy": {
            "reference": f"mpmath dps {MPMATH_DPS}",
            "logspaced_domain": accuracy_rows(logspaced_domain(), domain="logspaced |z| sweep"),
        },
    }

    if not include_runtime:
        return report

    grid, sample_coordinates = hst_grid_and_samples()
    samples, domain_summary = mge_domain_samples(grid)
    report["accuracy"]["mge_domain"] = accuracy_rows(samples, domain="gNFW MGE-30 hst inputs")
    report["accuracy"]["mge_domain_summary"] = domain_summary
    report["clamp"] = clamp_bias_rows(sample_coordinates, grid)
    report["clamp"]["q_sweep_gnfw_sph"] = clamp_q_sweep(sample_coordinates)
    report["clamp"]["q_sweep_gaussian"] = clamp_q_sweep_gaussian(sample_coordinates)
    report["deflection_jacobian"] = deflection_jacobian_rows(grid)
    report["centre_transect"] = centre_transect(grid)
    report["likelihood_transect"] = likelihood_transect()
    report["vmap_staticness"] = vmap_ell_comps_staticness()
    report["cost"] = {
        "block_shape": list(mge_domain_block(grid).shape),
        "omp_num_threads": _environment("OMP_NUM_THREADS"),
        "rows": cost_rows(mge_domain_block(grid)),
    }
    report["environment"] = _environment_block()
    return report


def _environment(name: str) -> str | None:
    import os

    return os.environ.get(name)


def _environment_block() -> dict:
    import platform

    import jax

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_device": str(jax.devices()[0]),
        "numpy": np.__version__,
        "omp_num_threads": _environment("OMP_NUM_THREADS"),
    }


def plot_report(report: dict, output: Path) -> None:
    """Two panels: accuracy over the input domain, and the transect gradient."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    rows = report["accuracy"]["logspaced_domain"]
    names = [row["routine"] for row in rows]
    errors = [max(row["max_relative_error"], 1.0e-17) for row in rows]
    axes[0].bar(range(len(names)), errors, color="#4C72B0")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([name.split(" ")[0] for name in names], rotation=20, fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max relative error vs mpmath")
    axes[0].set_title("Faddeeva accuracy (log-spaced |z| sweep)")

    transect = report.get("centre_transect")
    if transect is not None:
        centres = np.asarray(transect["series"]["centre_x"])
        gradient = np.asarray(transect["series"]["autodiff_gradient"], dtype=float)
        changed = np.asarray(
            [
                np.nan if value is None else value
                for value in transect["series"]["label_change_fraction"]
            ],
            dtype=float,
        )
        axes[1].plot(centres, gradient, linewidth=0.8, color="#4C72B0", label="jax.grad")
        twin = axes[1].twinx()
        twin.plot(centres, changed, linewidth=0.6, color="#DD8452", label="label churn")
        twin.set_ylabel("fraction of w-arguments changing region")
        axes[1].set_xlabel("centre_x (arcsec)")
        axes[1].set_ylabel("d(sum alpha^2)/d(centre_x)")
        axes[1].set_title("Deflection-gradient transect")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "runtime legs not run", ha="center", va="center")

    figure.suptitle("component.mge -- Faddeeva seams and the spherical clamp")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def json_safe(value, path: str = "", non_finite: list[str] | None = None):
    """Replace non-finite floats with ``None`` so the payload is valid JSON.

    Every substitution is reported rather than silently swallowed: a NaN in a
    measurement is a measurement that did not happen, and the note has to say so.
    """

    if non_finite is None:
        non_finite = []
    if isinstance(value, dict):
        return {key: json_safe(item, f"{path}.{key}", non_finite) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, f"{path}[{index}]", non_finite) for index, item in enumerate(value)]
    if isinstance(value, float) and not np.isfinite(value):
        non_finite.append(path)
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-report",
        action="store_true",
        help="include the hst dataset, likelihood, vmap and cost probes and print tagged JSON",
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
    non_finite: list[str] = []
    report = json_safe(report, non_finite=non_finite)
    if non_finite:
        print(f"non-finite values replaced with null: {sorted(set(non_finite))[:20]}")
    payload = json.dumps(report, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        plot_report(report, args.output.with_suffix(".png"))
        print(f"wrote {args.output}")
    if args.ci_report:
        print(f"MGE_FADDEEVA_REPORT={payload[:2000]}")
    elif args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
