"""
JAX Profiling: PointSolver Vertex-Dedup A/B (point-source CPU campaign, phase 2)
================================================================================

An interleaved, in-process A/B of one library method —
``autoarray.structures.triangles.coordinate_array.CoordinateArrayTriangles
._vertices_and_indices`` — inside the production point-source likelihoods
(PyAutoArray #568).

The mechanism
-------------

``PointSolver._plane_triangles`` ray-traces ``triangles.vertices`` and rebuilds
the triangles with ``with_vertices``. On the JAX path ``vertices`` /
``indices`` come from ``_vertices_and_indices``, which (PyAutoArray main
``11b93476``) calls::

    jnp.unique(flat_triangles, axis=0, return_inverse=True,
               size=3 * N, equal_nan=True, fill_value=jnp.nan)

``jnp.unique`` needs a static output shape under ``jit``, so it is handed
``size=3N`` — exactly the row count of the input it deduplicates. The
"deduplicated" table therefore has 3N rows (the real unique vertices plus NaN
fill), the ray trace evaluates all 3N of them, and the dedup saves **zero**
deflection evaluations while costing a lexicographic sort of 3N fp64 rows. The
property is read twice per refinement step (``vertices`` and ``indices`` each
evaluate it), so the sort is traced twice per step; XLA may CSE the pair, and
the HLO ``sort`` count recorded below says what survived. (Each row also records
the StableHLO count, which counts sort *definitions* inside private functions the
lowering reuses, not call sites — the optimised-HLO count is the one to read.) The NumPy sibling
(``coordinate_array_np.py``) calls ``np.unique`` with no ``size=`` and genuinely
shrinks the table; it is not touched here. See
``results/notes/point_source_cpu_2026_09_17_reported/pointsolver_cpu_research_note.md``
§3 and §6 for the reported scratch evidence this cell re-measures properly.

The fix drops the unique: the vertex table is the flat ``(3N, 2)`` triangle
array and the index map is ``arange(3N).reshape(-1, 3)``. NaN padding rows stay
NaN through the ray trace and every ``Shape.mask`` rejects them, so the kept
triangle set — and hence the likelihood — should be unchanged. Whether it is
*bit*-identical is measured, not assumed.

The three routes
----------------

Each route swaps the class attribute for the duration of **tracing** only, via
:func:`route_injected` (pattern: ``scripts/misc/likelihood_breakdown/
library_solver_injection.py`` — signature check, call counters, restore by
identity):

- ``control`` — the PRE-fix body, carried verbatim in this file
  (:func:`_control_vertices_and_indices`, copied from PyAutoArray main
  ``11b93476``). Reproducible whatever library is installed.
- ``nodedup`` — the POST-fix body (:func:`_nodedup_vertices_and_indices`).
- ``library`` — no substitution: whatever the installed PyAutoArray does. The
  route is wrapped only to count calls. It **self-labels**: the ``sort`` op count
  of its lowered / compiled program is compared with the other two routes and
  recorded as ``library_matches = control | nodedup | neither``.

A route whose substitute is never called while tracing is a FAIL, not a number.
If ``control`` and ``nodedup`` lower to the same number of sorts the injection
did not take and the cell stops before timing anything.

The measurement trap
--------------------

``jax`` caches traced jaxprs on function *identity*. The 2026-09-17 scratch run
first reported a speed-up with byte-identical HLO because ``jax.jit(same_fn)``
re-served the pre-patch trace. Here every route lowers a **fresh** closure built
by a factory (fresh ``PointSolver`` and ``AnalysisPoint`` too) and calls
``jax.clear_caches()`` before lowering. Once lowered and compiled, an executable
no longer depends on the patch, so all three routes' executables coexist and the
timed calls run with nothing patched.

The A/B protocol
----------------

Rows (``--models``): ``simple`` = the phase-1 harness model, dataset and solver
(``scripts/point_source/likelihood_breakdown/image_plane.py``: SIE lens,
``PointSolver.for_grid`` 100x100 @ 0.2", precision 0.001") with the fused
``FitPositionsImagePairAllSolved`` (``solved``) and ``FitPositionsImagePairAll``
(``plain``) likelihoods, plus a ``vmap`` batch row of the solved likelihood
(``--vmap-batch``); ``cluster`` = the cluster cell's two-source 13-component
model (``scripts/cluster/likelihood_breakdown/image_plane.py`` step 5; solver
200x200 @ 0.7", precision 0.01") with fused ``FitPositionsImagePairRepeat``
(``plain``) and ``FitPositionsImagePairRepeatSolved`` (``solved``) summed over
both systems.

Each compiled likelihood takes the model's **physical parameter vector** and
builds the instance inside the trace (``instance_from_vector(xp=jnp)``, as
``autofit``'s JAX ``Fitness`` does), so every parameter is a runtime argument.
The vectors are a fixed-seed stream drawn from the priors around the fiducial
(unit cube ``U(0.25, 0.75)``; entry 0 is the prior-median fiducial itself), and
the same stream feeds every route. Every timed call is ``block_until_ready``'d.

Per row: compile each route once, warm each ≥ 3 calls, then ``--rounds``
round-robin rounds, the route order rotating each round, ``--calls`` calls per
route per round, walking the same instance stream. Per route: median, p10, p90,
min, max of the per-call ms; the median ratio ``control / nodedup`` with a
bootstrap 90 % interval; the minimum detectable improvement is the control
row's ``(p90 - p10) / median``.

Correctness gates (recorded, never asserted — the architect decides): log
likelihood ``==`` across routes on every streamed instance (else max |Δ| and
relative Δ); solved image positions ``np.array_equal`` (NaN-aware) from a
separately compiled positions function per route; for the simple solved
likelihood, ``jax.grad`` w.r.t. the parameter vector ``allclose(rtol=1e-10,
atol=1e-12)`` and finite, and ``vmap`` results equal to the per-instance
results. The cluster rows carry the likelihood and position gates only: a
cluster gradient compile is minutes per route and phase 2 does not need it.

Output
------

``results/breakdown/point_source/vertex_dedup_ab_<config_name>.{json,png}``. The
JSON deliberately carries no top-level ``autolens_version`` (versions live under
``library_versions``): this is a note-backed A/B, and
``scripts/misc/tooling/build_readme.py`` only auto-tables config-tagged
breakdown JSONs whose ``autolens_version`` parses — which would render a
step-sum row this cell has no meaning for.
"""

import os
import sys
import time
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_MISC = str(_ROOT / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import contextlib
import hashlib
import inspect
import json
import re

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib
import numpy as np

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from likelihood_breakdown.provenance import source_revisions, thread_environment  # noqa: E402
from likelihood_breakdown.timing import block  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    machine_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

_WALL_START = time.perf_counter()
_LOADAVG_START = os.getloadavg()

_cli = parse_profile_cli()
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--rounds", type=int, default=None)
_cell_parser.add_argument("--calls", type=int, default=None)
_cell_parser.add_argument("--vmap-batch", type=int, default=4)
_cell_parser.add_argument("--n-stream", type=int, default=16)
_cell_parser.add_argument("--warm", type=int, default=3)
_cell_parser.add_argument("--gate-instances", type=int, default=3)
_cell_parser.add_argument("--seed", type=int, default=568)
_cell_parser.add_argument("--models", default=None, help="comma list of simple,cluster")
_cell_parser.add_argument(
    "--quick", action="store_true", help="rounds=5, calls=3, simple model only"
)
_args = _cli.parse_cell_args(_cell_parser)

QUICK = bool(_args.quick)
N_ROUNDS = _args.rounds if _args.rounds is not None else (5 if QUICK else 20)
N_CALLS = _args.calls if _args.calls is not None else (3 if QUICK else 5)
N_STREAM = _args.n_stream
N_WARM = max(3, _args.warm)
N_GATE = max(3, _args.gate_instances)
VMAP_BATCH = _args.vmap_batch
SEED = _args.seed
MODELS = tuple(
    m.strip()
    for m in (_args.models or ("simple" if QUICK else "simple,cluster")).split(",")
    if m.strip()
)
for _m in MODELS:
    if _m not in ("simple", "cluster"):
        raise SystemExit(f"--models: unknown model {_m!r} (expected simple / cluster)")
if N_ROUNDS < 1 or N_CALLS < 1 or VMAP_BATCH < 1:
    raise SystemExit("--rounds, --calls and --vmap-batch must be positive")

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autoarray.structures.triangles import coordinate_array as _coordinate_array  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402

ROUTES = ("control", "nodedup", "library")
TARGET_DOTTED = (
    "autoarray.structures.triangles.coordinate_array.CoordinateArrayTriangles._vertices_and_indices"
)
#: PyAutoArray main commit the ``control`` body below was copied from, verbatim.
CONTROL_SOURCE_SHA = "11b93476"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
GRAD_RTOL = 1.0e-10
GRAD_ATOL = 1.0e-12


# ---------------------------------------------------------------------------
# The two substituted bodies
# ---------------------------------------------------------------------------


def _control_vertices_and_indices(self):
    """PRE-fix ``_vertices_and_indices`` — verbatim from PyAutoArray ``11b93476``."""
    import jax.numpy as jnp

    flat_triangles = self.triangles.reshape(-1, 2)
    vertices, inverse_indices = jnp.unique(
        flat_triangles,
        axis=0,
        return_inverse=True,
        size=3 * self.coordinates.shape[0],
        equal_nan=True,
        fill_value=jnp.nan,
    )

    nan_mask = jnp.isnan(vertices).any(axis=1)
    inverse_indices = jnp.where(nan_mask[inverse_indices], -1, inverse_indices)

    indices = inverse_indices.reshape(-1, 3)
    return vertices, indices


def _nodedup_vertices_and_indices(self):
    """POST-fix ``_vertices_and_indices``: the flat 3N table and an ``arange`` index map."""
    import jax.numpy as jnp

    flat = self.triangles.reshape(-1, 2)
    indices = jnp.arange(flat.shape[0]).reshape(-1, 3)
    return flat, indices


_ROUTE_BODIES = {
    "control": _control_vertices_and_indices,
    "nodedup": _nodedup_vertices_and_indices,
    "library": None,
}


@contextlib.contextmanager
def route_injected(route: str):
    """Swap ``CoordinateArrayTriangles._vertices_and_indices`` for *route*.

    Scoped and restored by identity (checked on exit). Yields ``{"calls": n}``,
    the number of times the route's property body ran — i.e. while tracing — so
    the caller can refuse a route whose substitute never fired. ``library`` wraps
    the installed property only to count; it computes exactly what it did.
    """
    cls = _coordinate_array.CoordinateArrayTriangles
    original = cls.__dict__["_vertices_and_indices"]
    if not isinstance(original, property):
        raise TypeError(f"{TARGET_DOTTED} is no longer a property: {type(original).__name__}")
    params = list(inspect.signature(original.fget).parameters)
    if params != ["self"]:
        raise TypeError(
            f"{TARGET_DOTTED} now takes {params}; the substituted bodies take (self) only. "
            "The library changed under the injection — update this harness, not the library."
        )
    body = _ROUTE_BODIES[route] or original.fget
    counts = {"calls": 0}

    def counted(self):
        counts["calls"] += 1
        return body(self)

    cls._vertices_and_indices = property(counted)
    try:
        yield counts
    finally:
        cls._vertices_and_indices = original
        if cls.__dict__["_vertices_and_indices"] is not original:
            raise RuntimeError(f"{TARGET_DOTTED} was not restored by identity")


# ---------------------------------------------------------------------------
# Structural readouts
# ---------------------------------------------------------------------------

_STABLEHLO_SORT = re.compile(r"stablehlo\.sort\b")
_HLO_SORT_LINE = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(.+?)\s+sort\(", re.M)


def _sort_readout(lowered, compiled) -> dict:
    stablehlo = lowered.as_text()
    hlo = compiled.as_text()
    shapes = [m.group(1) for m in _HLO_SORT_LINE.finditer(hlo)]
    return {
        "hlo_sort_count": len(shapes),
        "stablehlo_sort_count": len(_STABLEHLO_SORT.findall(stablehlo)),
        "hlo_sort_shapes": shapes,
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest()[:16],
        "hlo_lines": hlo.count("\n"),
    }


def _flops(compiled):
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list | tuple):
            cost = cost[0] if cost else {}
        value = (cost or {}).get("flops")
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001 — cost analysis is optional on some backends
        return f"unavailable: {type(exc).__name__}"


def _compile_route(route: str, factory, example, *, label: str) -> dict:
    """Lower a FRESH closure under *route* after ``jax.clear_caches()``; compile it."""
    jax.clear_caches()
    fn = factory()
    with route_injected(route) as counts:
        t0 = time.perf_counter()
        lowered = jax.jit(fn).lower(example)
        lower_s = time.perf_counter() - t0
    if route != "library" and counts["calls"] == 0:
        raise RuntimeError(
            f"FAIL [{label}/{route}]: the substituted {TARGET_DOTTED} never ran while tracing, "
            "so this route would time the library's own code under the wrong name."
        )
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    first = block(compiled(example))
    first_call_s = time.perf_counter() - t0
    readout = _sort_readout(lowered, compiled)
    print(
        f"  [{label}/{route}] lower {lower_s:6.1f} s  compile {compile_s:6.1f} s  "
        f"first {first_call_s * 1e3:8.2f} ms  trace_calls {counts['calls']:3d}  "
        f"sorts {readout['hlo_sort_count']} (stablehlo {readout['stablehlo_sort_count']})"
    )
    return {
        "executable": compiled,
        "first": first,
        "record": {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "first_call_s": float(first_call_s),
            "trace_calls": int(counts["calls"]),
            "flops": _flops(compiled),
            **readout,
        },
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _stats_ms(seconds) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1.0e3
    return {
        "n": int(ms.size),
        "median_ms": float(np.median(ms)),
        "p10_ms": float(np.percentile(ms, 10)),
        "p90_ms": float(np.percentile(ms, 90)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "mean_ms": float(ms.mean()),
    }


def _median_ratio(numerator, denominator, seed: int) -> dict:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        boots[i] = np.median(rng.choice(num, num.size)) / np.median(rng.choice(den, den.size))
    return {
        "ratio": float(np.median(num) / np.median(den)),
        "ci90_low": float(np.percentile(boots, 5)),
        "ci90_high": float(np.percentile(boots, 95)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def _array_delta(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Equal entries (including matching NaN / inf padding) contribute zero; an
    # inf-vs-finite mismatch reports inf, never a NaN that nanmax would hide.
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.where(same, 0.0, np.abs(a - b))
        diff = np.where(np.isnan(diff), np.inf, diff)
        scale = np.where(same, 1.0, np.maximum(np.abs(a), np.abs(b)))
        rel = np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), diff)
    return {
        "bit_identical": bool(np.array_equal(a, b, equal_nan=True)),
        "max_abs_delta": float(np.nanmax(diff)) if diff.size else 0.0,
        "max_rel_delta": float(np.nanmax(rel)) if rel.size else 0.0,
        "nan_pattern_equal": bool(np.array_equal(np.isnan(a), np.isnan(b))),
    }


def _pairs(values_by_route: dict) -> dict:
    return {
        f"{a}_vs_{b}": _array_delta(values_by_route[a], values_by_route[b])
        for a, b in (("control", "nodedup"), ("control", "library"), ("nodedup", "library"))
    }


# ---------------------------------------------------------------------------
# Instance streams
# ---------------------------------------------------------------------------


def _vector_stream(model, n: int, seed: int) -> np.ndarray:
    """Fixed-seed physical vectors: entry 0 the prior medians, the rest U(0.25,0.75) draws."""
    rng = np.random.default_rng(seed)
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for _ in range(n - 1):
        unit = rng.uniform(0.25, 0.75, size=model.prior_count)
        vectors.append(np.asarray(model.vector_from_unit_vector(unit), dtype=float))
    return np.stack(vectors)


# ---------------------------------------------------------------------------
# The two models
# ---------------------------------------------------------------------------


def _simple_setup() -> dict:
    """Phase-1 harness model, dataset and solver (image_plane.py)."""
    instrument = "simple"
    dataset_path = _ROOT / "dataset" / "point_source" / instrument
    auto_simulate_if_missing(
        dataset_path, dataset_type="point_source", instrument=instrument, workspace_root=_ROOT
    )
    dataset = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")

    def mass_model():
        mass = af.Model(al.mp.Isothermal)
        mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
        mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
        mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
        mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
        mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
        return mass

    def model(solved: bool):
        lens = af.Model(al.Galaxy, redshift=0.5, mass=mass_model())
        if solved:
            point = af.Model(al.ps.PointSolved)
        else:
            point = af.Model(al.ps.PointFlux)
            point.centre.centre_0 = af.GaussianPrior(mean=0.07, sigma=0.005)
            point.centre.centre_1 = af.GaussianPrior(mean=0.07, sigma=0.005)
        source = af.Model(al.Galaxy, redshift=1.0, point_0=point)
        return af.Collection(galaxies=af.Collection(lens=lens, source=source))

    def solver():
        return al.PointSolver.for_grid(
            grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2),
            pixel_scale_precision=0.001,
            magnification_threshold=0.1,
        )

    fit_cls = {
        "solved": al.FitPositionsImagePairAllSolved,
        "plain": al.FitPositionsImagePairAll,
    }
    return {
        "datasets": [dataset],
        "models": {"solved": model(True), "plain": model(False)},
        "fit_cls": fit_cls,
        "solver": solver,
        "configuration": {
            "dataset": "dataset/point_source/simple/point_dataset_positions_only.json",
            "solver_grid_shape": [100, 100],
            "solver_grid_pixel_scale": 0.2,
            "solver_pixel_scale_precision": 0.001,
            "magnification_threshold": 0.1,
            "likelihoods": {
                "solved": "FitPositionsImagePairAllSolved",
                "plain": "FitPositionsImagePairAll",
            },
        },
    }


def _cluster_setup() -> dict:
    """The cluster cell's two-source 13-component model (image_plane.py step 5)."""
    dataset_path = _ROOT / "dataset" / "cluster" / "simple"
    auto_simulate_if_missing(
        dataset_path, dataset_type="cluster", instrument="simple", workspace_root=_ROOT
    )
    dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
    redshift_lens = 0.5
    main_lens_params = [((0.0, 0.0), 8.0, 20.0, 3.0), ((10.0, 8.0), 5.0, 12.0, 1.2)]
    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    b0_ref, rs_ref, ra_scaling, exponent = 0.12, 10.0, 0.1, 0.5
    lum_ref = max(scaling_table.luminosities)
    z_source_max = max(float(d.redshift) for d in dataset_list)

    # Step 1 of the cluster cell: back-trace the observed images through the
    # fiducial tracer and take each system's centroid as its model source centre.
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=redshift_lens,
                mass=al.mp.dPIEMassB0Sph(centre=centre, ra=ra, rs=rs, b0=b0),
            )
            for centre, ra, rs, b0 in main_lens_params
        ]
        + [
            al.Galaxy(
                redshift=redshift_lens,
                mass=al.mp.dPIEMassB0Sph(
                    centre=tuple(centre),
                    ra=ra_scaling,
                    rs=rs_ref * (lum / lum_ref) ** exponent,
                    b0=b0_ref * (lum / lum_ref) ** exponent,
                ),
            )
            for centre, lum in zip(scaling_table.centres, scaling_table.luminosities)
        ]
        + [
            al.Galaxy(
                redshift=redshift_lens,
                dark=al.mp.NFWMCRLudlowSph(
                    centre=(0.0, 0.0),
                    mass_at_200=10**15.3,
                    redshift_object=redshift_lens,
                    redshift_source=z_source_max,
                ),
            )
        ]
        + [
            al.Galaxy(redshift=float(d.redshift), **{d.name: al.ps.Point(centre=(0.0, 0.0))})
            for d in dataset_list
        ]
    )
    source_centres = []
    for d in dataset_list:
        plane_index = tracer.plane_index_via_redshift_from(redshift=float(d.redshift))
        traced = np.asarray(
            tracer.traced_grid_2d_list_from(
                grid=al.Grid2DIrregular(np.atleast_2d(np.asarray(d.positions)))
            )[plane_index]
        )
        source_centres.append(tuple(traced.mean(axis=0)))

    def gaussian(mean, fraction=0.01, floor=0.01):
        return af.GaussianPrior(mean=float(mean), sigma=max(abs(float(mean)) * fraction, floor))

    def model(solved: bool):
        galaxies = {}
        for i, (centre, ra, rs, b0) in enumerate(main_lens_params):
            mass = af.Model(al.mp.dPIEMassB0Sph)
            mass.centre.centre_0 = gaussian(centre[0])
            mass.centre.centre_1 = gaussian(centre[1])
            mass.ra = gaussian(ra)
            mass.rs = gaussian(rs)
            mass.b0 = gaussian(b0)
            galaxies[f"main_{i}"] = af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
        scaling_b0 = gaussian(b0_ref, floor=0.001)
        for i, (centre, lum) in enumerate(zip(scaling_table.centres, scaling_table.luminosities)):
            ratio = (lum / lum_ref) ** exponent
            mass = af.Model(al.mp.dPIEMassB0Sph)
            mass.centre = tuple(centre)
            mass.ra = ra_scaling
            mass.rs = rs_ref * ratio
            mass.b0 = scaling_b0 * ratio
            galaxies[f"scaling_{i}"] = af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
        dark = af.Model(
            al.mp.NFWMCRLudlowSph,
            mass_at_200=10**15.3,
            redshift_object=redshift_lens,
            redshift_source=z_source_max,
        )
        dark.centre.centre_0 = gaussian(0.0)
        dark.centre.centre_1 = gaussian(0.0)
        galaxies["host_halo"] = af.Model(al.Galaxy, redshift=redshift_lens, dark=dark)
        for i, (d, centre) in enumerate(zip(dataset_list, source_centres)):
            if solved:
                point = af.Model(al.ps.PointSolved)
            else:
                point = af.Model(al.ps.Point)
                point.centre.centre_0 = gaussian(centre[0])
                point.centre.centre_1 = gaussian(centre[1])
            galaxies[f"source_{i}"] = af.Model(
                al.Galaxy, redshift=float(d.redshift), **{d.name: point}
            )
        return af.Collection(galaxies=af.Collection(**galaxies))

    def solver():
        return al.PointSolver.for_grid(
            grid=al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.7),
            pixel_scale_precision=0.01,
            use_jax=True,
        )

    fit_cls = {
        "solved": al.FitPositionsImagePairRepeatSolved,
        "plain": al.FitPositionsImagePairRepeat,
    }
    return {
        "datasets": list(dataset_list),
        "models": {"solved": model(True), "plain": model(False)},
        "fit_cls": fit_cls,
        "solver": solver,
        "configuration": {
            "dataset": "dataset/cluster/simple/point_datasets.csv",
            "n_systems": len(dataset_list),
            "n_mass_profiles": len(main_lens_params) + len(scaling_table.luminosities) + 1,
            "solver_grid_shape": [200, 200],
            "solver_grid_pixel_scale": 0.7,
            "solver_pixel_scale_precision": 0.01,
            "source_centres_from": "back-traced observed-image centroids (cluster cell step 1)",
            "likelihoods": {
                "solved": "FitPositionsImagePairRepeatSolved (summed over systems)",
                "plain": "FitPositionsImagePairRepeat (summed over systems)",
            },
        },
    }


def _likelihood_factory(setup: dict, variant: str):
    """A factory of FRESH ``vector -> log L`` closures (fresh solver + analyses per call)."""
    model = setup["models"][variant]

    def factory():
        solver = setup["solver"]()
        analyses = [
            al.AnalysisPoint(
                dataset=d, solver=solver, fit_positions_cls=setup["fit_cls"][variant], use_jax=True
            )
            for d in setup["datasets"]
        ]

        def log_likelihood(vector):
            instance = model.instance_from_vector(vector=vector, xp=jnp)
            total = 0.0
            for analysis in analyses:
                total = total + analysis.log_likelihood_function(instance=instance)
            return total

        return log_likelihood

    return factory


def _positions_factory(setup: dict, variant: str):
    """A factory of FRESH ``vector -> tuple(model positions per system)`` closures."""
    model = setup["models"][variant]

    def factory():
        solver = setup["solver"]()
        analyses = [
            al.AnalysisPoint(
                dataset=d, solver=solver, fit_positions_cls=setup["fit_cls"][variant], use_jax=True
            )
            for d in setup["datasets"]
        ]

        def positions(vector):
            instance = model.instance_from_vector(vector=vector, xp=jnp)
            out = []
            for analysis in analyses:
                model_data = analysis.fit_from(instance=instance).positions.model_data
                out.append(jnp.asarray(getattr(model_data, "array", model_data)))
            return tuple(out)

        return positions

    return factory


# ---------------------------------------------------------------------------
# One interleaved timing row
# ---------------------------------------------------------------------------


def _timed_call(executable, argument):
    t0 = time.perf_counter()
    out = block(executable(argument))
    return time.perf_counter() - t0, out


def _run_row(label: str, factory, stream: np.ndarray, *, batch: int | None = None) -> dict:
    """Compile three routes, warm, interleave; return timings, values and structure."""
    print(f"\n--- row {label} ---")
    if batch is None:
        arguments = [jnp.asarray(v) for v in stream]
        instance_ids = [[i] for i in range(len(stream))]
    else:
        n_batches = max(1, len(stream) // batch)
        instance_ids = [
            [(b * batch + k) % len(stream) for k in range(batch)] for b in range(n_batches)
        ]
        arguments = [jnp.asarray(stream[ids]) for ids in instance_ids]
    compiled = {
        route: _compile_route(route, factory, arguments[0], label=label) for route in ROUTES
    }
    executables = {route: compiled[route]["executable"] for route in ROUTES}

    warm_s = {route: [] for route in ROUTES}
    for route in ROUTES:
        for w in range(N_WARM):
            dt, _ = _timed_call(executables[route], arguments[w % len(arguments)])
            warm_s[route].append(dt)

    times = {route: [] for route in ROUTES}
    per_round = {route: [] for route in ROUTES}
    values = {route: {} for route in ROUTES}
    deterministic = {route: True for route in ROUTES}
    call_index = 0
    for r in range(N_ROUNDS):
        order = ROUTES[r % len(ROUTES) :] + ROUTES[: r % len(ROUTES)]
        start = call_index
        for route in order:
            round_times = []
            for c in range(N_CALLS):
                k = (start + c) % len(arguments)
                dt, out = _timed_call(executables[route], arguments[k])
                round_times.append(dt)
                value = np.asarray(out, dtype=float)
                if k in values[route]:
                    if not np.array_equal(values[route][k], value, equal_nan=True):
                        deterministic[route] = False
                else:
                    values[route][k] = value
            times[route].extend(round_times)
            per_round[route].append(float(np.median(round_times)))
        call_index = start + N_CALLS

    seen = sorted(set.intersection(*(set(values[r]) for r in ROUTES)))
    value_arrays = {route: np.stack([values[route][k] for k in seen]) for route in ROUTES}
    stats = {route: _stats_ms(times[route]) for route in ROUTES}
    control = stats["control"]
    paired = np.asarray(per_round["control"]) / np.asarray(per_round["nodedup"])
    row = {
        "label": label,
        "batch": batch,
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "route_order": "round-robin, start rotated each round",
        "per_call_ms": {route: [t * 1e3 for t in times[route]] for route in ROUTES},
        "warm_ms": {route: [t * 1e3 for t in warm_s[route]] for route in ROUTES},
        "stats": stats,
        "ratio_control_over_nodedup": _median_ratio(
            times["control"], times["nodedup"], BOOTSTRAP_SEED
        ),
        "ratio_library_over_nodedup": _median_ratio(
            times["library"], times["nodedup"], BOOTSTRAP_SEED + 1
        ),
        "ratio_control_over_library": _median_ratio(
            times["control"], times["library"], BOOTSTRAP_SEED + 2
        ),
        "paired_round_ratio_control_over_nodedup": {
            "median": float(np.median(paired)),
            "p10": float(np.percentile(paired, 10)),
            "p90": float(np.percentile(paired, 90)),
        },
        "minimum_detectable_improvement": float(
            (control["p90_ms"] - control["p10_ms"]) / control["median_ms"]
        ),
        "compile": {route: compiled[route]["record"] for route in ROUTES},
        "instances_compared": [instance_ids[k] for k in seen],
        "values": {route: value_arrays[route].tolist() for route in ROUTES},
        "deterministic_within_route": deterministic,
        "likelihood_gate": _pairs(value_arrays),
    }
    if batch is None:
        row["fiducial_log_likelihood"] = {
            route: float(values[route][0]) for route in ROUTES if 0 in values[route]
        }
    print(
        "  median ms  "
        + "  ".join(f"{route} {stats[route]['median_ms']:9.3f}" for route in ROUTES)
        + f"   control/nodedup {row['ratio_control_over_nodedup']['ratio']:.3f} "
        f"[{row['ratio_control_over_nodedup']['ci90_low']:.3f}, "
        f"{row['ratio_control_over_nodedup']['ci90_high']:.3f}]"
    )
    return row


def _positions_gate(label: str, factory, stream: np.ndarray) -> dict:
    print(f"\n--- positions gate {label} ---")
    arguments = [jnp.asarray(v) for v in stream[:N_GATE]]
    per_route = {}
    compile_records = {}
    for route in ROUTES:
        compiled = _compile_route(route, factory, arguments[0], label=f"{label}_positions")
        compile_records[route] = compiled["record"]
        per_route[route] = [
            [np.asarray(p, dtype=float) for p in block(compiled["executable"](a))]
            for a in arguments
        ]

    def flat(route):
        return np.concatenate([np.ravel(p) for inst in per_route[route] for p in inst])

    pairs = {}
    for a, b in (("control", "nodedup"), ("control", "library")):
        delta = _array_delta(flat(a), flat(b))
        delta["array_equal_nan_aware"] = delta.pop("bit_identical")
        pairs[f"{a}_vs_{b}"] = delta
    return {
        "n_instances": len(arguments),
        "compile": compile_records,
        "positions": {
            route: [[p.tolist() for p in inst] for inst in per_route[route]] for route in ROUTES
        },
        "finite_positions": {
            route: [[int(np.isfinite(p).all(axis=-1).sum()) for p in inst] for inst in rows]
            for route, rows in per_route.items()
        },
        "pairs": pairs,
    }


def _grad_gate(label: str, factory, stream: np.ndarray) -> dict:
    print(f"\n--- gradient gate {label} ---")
    arguments = [jnp.asarray(v) for v in stream[:N_GATE]]

    def grad_factory():
        return jax.grad(factory())

    grads = {}
    records = {}
    for route in ROUTES:
        compiled = _compile_route(route, grad_factory, arguments[0], label=f"{label}_grad")
        records[route] = compiled["record"]
        grads[route] = np.stack(
            [np.asarray(block(compiled["executable"](a)), dtype=float) for a in arguments]
        )
    pairs = {}
    for a, b in (("control", "nodedup"), ("control", "library")):
        delta = _array_delta(grads[a], grads[b])
        delta["allclose"] = bool(np.allclose(grads[a], grads[b], rtol=GRAD_RTOL, atol=GRAD_ATOL))
        pairs[f"{a}_vs_{b}"] = delta
    return {
        "n_instances": len(arguments),
        "rtol": GRAD_RTOL,
        "atol": GRAD_ATOL,
        "compile": records,
        "all_finite": {route: bool(np.isfinite(g).all()) for route, g in grads.items()},
        "gradients": {route: g.tolist() for route, g in grads.items()},
        "pairs": pairs,
    }


def _vmap_gate(vmap_row: dict, scalar_row: dict) -> dict:
    """``vmap`` batch values vs the scalar row's per-instance values, per route and across."""
    scalar_values = {
        route: dict(
            zip(
                [ids[0] for ids in scalar_row["instances_compared"]],
                np.asarray(scalar_row["values"][route]).ravel(),
            )
        )
        for route in ROUTES
    }
    within = {}
    for route in ROUTES:
        batched, reference = [], []
        for ids, vals in zip(vmap_row["instances_compared"], vmap_row["values"][route]):
            for i, v in zip(ids, np.ravel(vals)):
                if i in scalar_values[route]:
                    batched.append(v)
                    reference.append(scalar_values[route][i])
        delta = _array_delta(batched, reference)
        delta["allclose_rtol_1e-12"] = bool(
            np.allclose(batched, reference, rtol=1e-12, atol=0.0, equal_nan=True)
        )
        delta["n_compared"] = len(batched)
        within[route] = delta
    return {"vmap_vs_scalar_within_route": within, "across_routes": vmap_row["likelihood_gate"]}


def _library_label(rows: dict) -> dict:
    per_row = {}
    for label, row in rows.items():
        counts = {route: row["compile"][route]["hlo_sort_count"] for route in ROUTES}
        if counts["control"] == counts["nodedup"]:
            match = "injection_ineffective"
        elif counts["library"] == counts["control"]:
            match = "control"
        elif counts["library"] == counts["nodedup"]:
            match = "nodedup"
        else:
            match = "neither"
        hashes = {route: row["compile"][route]["hlo_sha256"] for route in ROUTES}
        per_row[label] = {
            "hlo_sort_count": counts,
            "stablehlo_sort_count": {
                route: row["compile"][route]["stablehlo_sort_count"] for route in ROUTES
            },
            "library_matches": match,
            "library_hlo_identical_to": [
                route for route in ("control", "nodedup") if hashes[route] == hashes["library"]
            ],
        }
    labels = {entry["library_matches"] for entry in per_row.values()}
    overall = labels.pop() if len(labels) == 1 else "inconsistent"
    return {"per_row": per_row, "library_matches": overall}


# ===========================================================================
# Run
# ===========================================================================

rows: dict[str, dict] = {}
gates: dict[str, dict] = {}
model_configurations: dict[str, dict] = {}
free_parameters: dict[str, int] = {}

for model_name in MODELS:
    setup = _simple_setup() if model_name == "simple" else _cluster_setup()
    model_configurations[model_name] = setup["configuration"]
    # Pytree registration, as the phase-1 cells and autofit's JAX Fitness do. It is
    # load-bearing for the gradient gate: without it, jax.grad of this likelihood
    # w.r.t. the physical vector comes back identically ZERO (forward values are
    # unaffected) — measured 2026-09-23 on PyAutoFit/PyAutoLens main.
    for _model in setup["models"].values():
        register_model_pytrees(_model)
    for variant in ("solved", "plain"):
        model = setup["models"][variant]
        free_parameters[f"{model_name}_{variant}"] = int(model.prior_count)
        stream = _vector_stream(model, N_STREAM, SEED + (0 if variant == "solved" else 1))
        factory = _likelihood_factory(setup, variant)
        label = f"{model_name}_{variant}"
        rows[label] = _run_row(label, factory, stream)
        if label == "simple_solved":
            # Fail fast: if the two substituted bodies lower to the same program
            # the injection did not take, and nothing below would mean anything.
            counts = {r: rows[label]["compile"][r]["hlo_sort_count"] for r in ROUTES}
            if counts["control"] == counts["nodedup"]:
                raise RuntimeError(
                    f"FAIL: control and nodedup lower with the same sort count ({counts}); "
                    "the injection did not change the traced program."
                )
        if variant == "solved":
            gates[f"{label}_positions"] = _positions_gate(
                label, _positions_factory(setup, variant), stream
            )
        if label == "simple_solved":
            gates[f"{label}_grad"] = _grad_gate(label, factory, stream)
            vlabel = f"{label}_vmap{VMAP_BATCH}"
            rows[vlabel] = _run_row(
                vlabel, lambda f=factory: jax.vmap(f()), stream, batch=VMAP_BATCH
            )
            gates[f"{vlabel}"] = _vmap_gate(rows[vlabel], rows[label])

library_label = _library_label(rows)


def _gate_summary() -> dict:
    """One flat verdict table: every gate, control vs nodedup and control vs library."""
    summary = {}
    for label, row in rows.items():
        for pair in ("control_vs_nodedup", "control_vs_library"):
            g = row["likelihood_gate"][pair]
            summary[f"{label}.log_likelihood.{pair}"] = {
                "pass": g["bit_identical"],
                "max_abs_delta": g["max_abs_delta"],
                "max_rel_delta": g["max_rel_delta"],
            }
    for label, gate in gates.items():
        if label.endswith("_positions"):
            for pair, g in gate["pairs"].items():
                summary[f"{label}.{pair}"] = {
                    "pass": g["array_equal_nan_aware"],
                    "max_abs_delta": g["max_abs_delta"],
                }
        elif label.endswith("_grad"):
            for pair, g in gate["pairs"].items():
                summary[f"{label}.{pair}"] = {
                    "pass": g["allclose"] and all(gate["all_finite"].values()),
                    "bit_identical": g["bit_identical"],
                    "max_abs_delta": g["max_abs_delta"],
                    "max_rel_delta": g["max_rel_delta"],
                }
        else:
            for route, g in gate["vmap_vs_scalar_within_route"].items():
                summary[f"{label}.vmap_vs_scalar.{route}"] = {
                    "pass": g["allclose_rtol_1e-12"],
                    "bit_identical": g["bit_identical"],
                    "max_abs_delta": g["max_abs_delta"],
                }
    return summary


gate_summary = _gate_summary()
bit_identical = all(
    rows[label]["likelihood_gate"][pair]["bit_identical"]
    for label in rows
    for pair in ("control_vs_nodedup", "control_vs_library")
)

config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_device_fp64"
)
_home = str(Path.home())


def _scrub(obj):
    """Replace the home directory in every string (no /home/<user> in a committed JSON)."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str) and _home and _home != "/":
        return obj.replace(_home, "~")
    return obj


device = device_info_dict()
if device.get("jax_compilation_cache_dir"):
    device["jax_compilation_cache_dir"] = (
        "<scrubbed>/" + Path(device["jax_compilation_cache_dir"]).name
    )

summary = {
    "cell": "vertex_dedup_ab",
    "issue": "PyAutoLabs/PyAutoArray#568",
    "config_name": config_name,
    "quick": QUICK,
    "models": list(MODELS),
    "library_versions": {
        "autolens": al.__version__,
        "autoarray": aa.__version__,
        "autofit": af.__version__,
    },
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    "source_revisions": source_revisions(_ROOT),
    "autoarray_imported_from": str(Path(aa.__file__).resolve().parent.parent),
    "target": TARGET_DOTTED,
    "control_source": {
        "repo": "PyAutoArray",
        "sha": CONTROL_SOURCE_SHA,
        "path": "autoarray/structures/triangles/coordinate_array.py",
        "body": inspect.getsource(_control_vertices_and_indices),
    },
    "nodedup_body": inspect.getsource(_nodedup_vertices_and_indices),
    "device": device,
    "jax_devices": [str(d) for d in jax.devices()],
    "machine": machine_info_dict(),
    "thread_environment": thread_environment(),
    "sched_affinity": len(os.sched_getaffinity(0)),
    "os_cpu_count": os.cpu_count(),
    "xla_flags": os.environ.get("XLA_FLAGS"),
    "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "loadavg_start": list(_LOADAVG_START),
    "loadavg_end": list(os.getloadavg()),
    "protocol": {
        "routes": list(ROUTES),
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "n_stream": N_STREAM,
        "n_gate_instances": N_GATE,
        "vmap_batch": VMAP_BATCH,
        "instance_seed": SEED,
        "instance_stream": "entry 0 = prior medians; rest = vector_from_unit_vector(U(0.25,0.75))",
        "argument": "physical parameter vector; instance_from_vector(xp=jnp) inside the trace",
        "cache_trap_guard": "fresh closure + fresh solver/analyses per route; jax.clear_caches()",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": "90%"},
        "cluster_gates": "log-likelihood and positions only (no cluster grad/vmap)",
        "pytrees": "autofit.jax.register_model(model) for every model before any trace",
    },
    "free_parameters": free_parameters,
    "model_configuration": model_configurations,
    "library_label": library_label,
    "library_matches": library_label["library_matches"],
    "bit_identical": bit_identical,
    "gate_summary": gate_summary,
    "rows": rows,
    "gates": gates,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source",
    default_basename="vertex_dedup_ab_local",
    cell="vertex_dedup_ab",
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2))

# ---------------------------------------------------------------------------
# PNG: per-row boxplots, control vs nodedup vs library
# ---------------------------------------------------------------------------
colours = {"control": "#C44E52", "nodedup": "#4C72B0", "library": "#8C8C8C"}
fig, axes = plt.subplots(
    nrows=1, ncols=len(rows), figsize=(4.2 * len(rows), 5.0), constrained_layout=True
)
axes = np.atleast_1d(axes)
for ax, (label, row) in zip(axes, rows.items()):
    data = [row["per_call_ms"][route] for route in ROUTES]
    box = ax.boxplot(data, whis=(10, 90), showfliers=True, patch_artist=True)
    for patch, route in zip(box["boxes"], ROUTES):
        patch.set_facecolor(colours[route])
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(ROUTES) + 1))
    ax.set_xticklabels(ROUTES)
    ax.set_ylabel("ms per call" + (f" (batch of {row['batch']})" if row["batch"] else ""))
    ratio = row["ratio_control_over_nodedup"]
    ax.set_title(
        f"{label}\ncontrol/nodedup {ratio['ratio']:.2f}x "
        f"[{ratio['ci90_low']:.2f}, {ratio['ci90_high']:.2f}]",
        fontsize=9,
    )
    ax.set_ylim(bottom=0)
fig.suptitle(
    f"PointSolver vertex-dedup A/B ({config_name}); library_matches="
    f"{library_label['library_matches']}; bit_identical={bit_identical}; "
    f"whiskers p10/p90",
    fontsize=10,
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 100)
print(f"VERTEX-DEDUP A/B  ({config_name}, rounds={N_ROUNDS}, calls/round={N_CALLS})")
print("=" * 100)
print(
    f"  {'row':<22} {'route':<8} {'median':>9} {'p10':>9} {'p90':>9} {'min':>9} {'max':>9}"
    f" {'sorts':>6} {'compile_s':>9}"
)
for label, row in rows.items():
    for route in ROUTES:
        s = row["stats"][route]
        c = row["compile"][route]
        print(
            f"  {label:<22} {route:<8} {s['median_ms']:9.3f} {s['p10_ms']:9.3f} {s['p90_ms']:9.3f}"
            f" {s['min_ms']:9.3f} {s['max_ms']:9.3f} {c['hlo_sort_count']:6d}"
            f" {c['compile_s']:9.1f}"
        )
    ratio = row["ratio_control_over_nodedup"]
    print(
        f"  {'':<22} control/nodedup {ratio['ratio']:.3f} "
        f"[90% {ratio['ci90_low']:.3f}, {ratio['ci90_high']:.3f}]  "
        f"MDI {row['minimum_detectable_improvement']:.3f}"
    )
print("-" * 100)
print(f"  library_matches: {library_label['library_matches']}")
print(f"  bit_identical (log L, all rows, control vs nodedup/library): {bit_identical}")
for name, verdict in gate_summary.items():
    if not verdict["pass"]:
        print(f"  GATE NOT PASSED: {name}: {verdict}")
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
