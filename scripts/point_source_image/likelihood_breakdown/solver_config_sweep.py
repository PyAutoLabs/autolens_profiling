"""
JAX Profiling: PointSolver Configuration Sweep (point-source CPU campaign, phase 4a)
===================================================================================

A measured sweep of the JAX ``PointSolver``'s configuration knobs on the production
single-source image-plane likelihood (``FitPositionsImagePairAllSolved``), with an
image-completeness gate on every configuration (autolens_profiling#314, phase 4a).
No library code is changed: every configuration is built through the public
``PointSolver.for_grid`` / ``PointSolver.for_limits_and_scale`` constructors, except
``MAX_CONTAINING_SIZE``, a PyAutoArray module constant that is patched for the
duration of **tracing** only (see :func:`max_containing_size`).

Why these knobs
---------------

The post-phase-3 re-baseline (RAL job 356365, Xeon 8490H, released code) split the
≈ 2.1 ms solved likelihood as: step 0 66 % (ray trace 46 %, containment 22 %, 23 283
triangles), refinement steps 1-7 only ≈ 15 % (≈ 0.045 ms each, latency-bound), β*
8 %, magnification filter 7 %, χ² 4 %. So the primary axes are the two that shrink
step 0:

- **extent** -- the half-width of the image-plane box the step-0 lattice tiles
  (production: the 100 x 100 x 0.2" grid, ±9.9");
- **initial scale** -- the step-0 triangle side. A coarser scale means fewer step-0
  rows and more refinement steps, ``n_steps = ceil(log2(scale / precision))``.

The final precision ``pixel_scale_precision`` is held at the production 0.001": it
sets the answer's precision, not the cost of finding it. Secondary blocks (a few
rows each): a *finer* step 0 (the phase-3 hand-off direction, expected to lose),
``MAX_CONTAINING_SIZE`` and ``neighbor_degree``, and -- clearly labelled, never the
headline -- a coarser ``pixel_scale_precision`` (a correctness trade).

Each configuration is its own timed route with its own row. No configuration's
speed is ever substituted into another's figure.

Completeness gate
-----------------

- **Prior draws** (the admissibility gate): ``--n-draws`` seeded draws of the
  point-source model's full prior (the solved model of ``image_plane.py``:
  Isothermal centre N(0, 0.005), θ_E N(1.6, 0.05), ell_comps N(0.0526, 0.01) /
  N(0, 0.01); the source centre is the likelihood's own solved β*, so the image
  positions are the solver's, not the data's). The likelihood is evaluated on the
  fixed dataset, so |Δ log L| applies.
- **Stress draws** (reported, not the admissibility gate): the model prior is narrow
  (every draw is a quad close to the data), so a second set draws lens and source
  from a deliberately broad box OUTSIDE the prior -- Isothermal centre U(-0.05,
  0.05), θ_E U(1.0, 2.0), ell_comps U(-0.2, 0.2), source centre U(-0.4, 0.4) -- and
  solves the plain (source-centre) route for doubles, quads and near-caustic
  sources. It shows where a knob stops being safe once the prior widens.
- **Reference** per draw: initial scale 0.05", extent ±12", ``MAX_CONTAINING_SIZE``
  60, precision 1e-4" (ten times finer than production, so the position error
  measures the configuration, not the reference; final triangle side 9.8e-5").
  A second, independent reference (0.07", ±11", 60, 1e-4") is solved on every draw
  too; its disagreement with the first is the reference's own error floor.
- Per configuration and draw, after the magnification filter: the raw finite
  position count, the **distinct image multiplicity** (positions merged within
  ``DISTINCT_TOL``), the nearest-neighbour position error of every matched image,
  and |Δ log L| against the reference and against the production default.
- **Admissible** = 100 % distinct-multiplicity agreement with the reference on the
  prior draws AND max position error ≤ ``ADMISSIBLE_POSITION_TOL`` = 0.002" (twice
  the nominal production precision; σ_pos = 0.05" in the dataset, so 0.002" is
  σ/25 and moves χ² by ≲ 2 δ r / σ² ≈ 0.016 per image at a 0.1" residual). The
  solver's returned positions are the means of the kept triangles of its LAST step,
  whose side is ``scale / 2**(n_steps - 1)`` (0.0016" for the production 0.2" /
  0.001"), so the production default itself sits at ≈ 0.0008-0.0009" error and a
  0.001" tolerance would reject configurations for rounding in ``ceil(log2(.))``.
- **Precision-equivalent** = the last-step triangle side is ≤ the control's. A
  coarser initial scale changes ``n_steps`` by the ceiling, so e.g. 0.5" / 0.001"
  ends on 0.00195" triangles, 25 % coarser than the control. ``best_admissible``
  requires both flags; ``best_admissible_any_precision`` only the first.

Timing
------

Phase-3 protocol: fresh closures and ``jax.clear_caches()`` per route, the
production default as the in-job control, interleaved round-robin rounds with the
start rotated each round, per-call medians, bootstrap 90 % CI of control / config,
``compile_s``, FLOPs (``cost_analysis``), XLA ``memory_analysis`` and ``vmap``
batches 1 / 4 / 16 for the control and the best admissible configuration. For each
distinct step-0 geometry the step-0 ray trace and ray trace + containment are timed
as their own jitted prefixes, so the containment share is recorded.
Finally, the NumPy solver (dynamic shapes, no ``MAX_CONTAINING_SIZE`` cap) counts how
many triangles contain β* at every step on the prior draws, for the control and the
best admissible geometries: the margin the production cap of 15 actually has.

Gates
-----

- The control's solved log L on the prior-median vector equals the fiducial
  ``7.743201200876812`` bit-exactly.
- ``jax.grad`` of the likelihood is finite and non-zero for every admissible
  configuration. ``autofit.jax.register_model(model)`` is load-bearing: without it
  the gradient is silently all-zero.
- The ``MAX_CONTAINING_SIZE`` patch is proved per route from the traced shapes (the
  padded solution has ``MAX_CONTAINING_SIZE`` rows).

Output
------

``results/breakdown/point_source_image/solver_config_sweep_<config_name>.{json,png}``.
A ``laptop_*`` / ``local_*`` config is load-inflated and is recorded
``quotable: false``. No top-level ``autolens_version``, so ``build_readme.py`` does
not auto-table it.
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
import json
import math

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
_cell_parser.add_argument("--n-draws", type=int, default=None, help="prior draws (gate)")
_cell_parser.add_argument("--n-stress", type=int, default=None, help="broad stress draws")
_cell_parser.add_argument("--configs", default=None, help="comma list of config names")
_cell_parser.add_argument("--n-stream", type=int, default=16)
_cell_parser.add_argument("--warm", type=int, default=3)
_cell_parser.add_argument("--grad-instances", type=int, default=3)
_cell_parser.add_argument("--vmap-batches", default="1,4,16")
_cell_parser.add_argument("--step0-rounds", type=int, default=None)
_cell_parser.add_argument("--seed", type=int, default=314)
_cell_parser.add_argument("--list-configs", action="store_true")
_cell_parser.add_argument(
    "--quick",
    action="store_true",
    help="laptop witness: rounds=3, calls=3, n-draws=20, n-stress=20",
)
_args = _cli.parse_cell_args(_cell_parser)

QUICK = bool(_args.quick)
N_ROUNDS = _args.rounds if _args.rounds is not None else (3 if QUICK else 20)
N_CALLS = _args.calls if _args.calls is not None else (3 if QUICK else 20)
N_DRAWS = _args.n_draws if _args.n_draws is not None else (20 if QUICK else 200)
N_STRESS = _args.n_stress if _args.n_stress is not None else (20 if QUICK else 200)
N_STEP0_ROUNDS = _args.step0_rounds if _args.step0_rounds is not None else (3 if QUICK else 10)
N_STREAM = _args.n_stream
N_WARM = max(3, _args.warm)
N_GRAD = max(1, _args.grad_instances)
SEED = _args.seed
VMAP_BATCHES = tuple(int(b) for b in _args.vmap_batches.split(",") if b.strip())
if N_ROUNDS < 1 or N_CALLS < 1 or N_DRAWS < 1 or N_STRESS < 0 or not VMAP_BATCHES:
    raise SystemExit("--rounds, --calls, --n-draws must be positive; --vmap-batches non-empty")

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autoarray.structures.triangles import array as _triangles_array  # noqa: E402
from autoarray.structures.triangles import coordinate_array as _coordinate_array  # noqa: E402
from autoarray.structures.triangles.shape import Point  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402
from autolens.point.solver import shape_solver as _shape_solver  # noqa: E402

FIDUCIAL_SOLVED_LOG_L = 7.743201200876812
PRODUCTION_PRECISION = 0.001
PRODUCTION_MCS = int(_triangles_array.MAX_CONTAINING_SIZE)
ADMISSIBLE_POSITION_TOL = 0.002
DISTINCT_TOL = 0.005
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
GRID_SHAPE = (100, 100)
GRID_PIXEL_SCALE = 0.2

# ---------------------------------------------------------------------------
# Configurations. extent None = the production grid's own limits (±9.9").
# ---------------------------------------------------------------------------


def _cfg(
    name,
    block,
    *,
    extent=None,
    scale=0.2,
    precision=PRODUCTION_PRECISION,
    mcs=None,
    neighbor_degree=1,
):
    return {
        "name": name,
        "block": block,
        "extent": extent,
        "scale": scale,
        "precision": precision,
        "max_containing_size": PRODUCTION_MCS if mcs is None else mcs,
        "neighbor_degree": neighbor_degree,
    }


CONFIGS = [
    _cfg("control", "control"),
    # (a) extent at the production scale
    _cfg("e6_s0.2", "extent", extent=6.0),
    _cfg("e4_s0.2", "extent", extent=4.0),
    _cfg("e3_s0.2", "extent", extent=3.0),
    _cfg("e2.5_s0.2", "extent", extent=2.5),
    # (b') coarser initial scale at the production extent
    _cfg("e9.9_s0.3", "scale", scale=0.3),
    _cfg("e9.9_s0.4", "scale", scale=0.4),
    _cfg("e9.9_s0.5", "scale", scale=0.5),
    _cfg("e9.9_s0.8", "scale", scale=0.8),
    # extent x scale
    _cfg("e4_s0.3", "combo", extent=4.0, scale=0.3),
    _cfg("e4_s0.4", "combo", extent=4.0, scale=0.4),
    _cfg("e4_s0.5", "combo", extent=4.0, scale=0.5),
    _cfg("e4_s0.8", "combo", extent=4.0, scale=0.8),
    _cfg("e3_s0.3", "combo", extent=3.0, scale=0.3),
    _cfg("e3_s0.4", "combo", extent=3.0, scale=0.4),
    _cfg("e3_s0.5", "combo", extent=3.0, scale=0.5),
    _cfg("e3_s0.8", "combo", extent=3.0, scale=0.8),
    _cfg("e2.5_s0.4", "combo", extent=2.5, scale=0.4),
    # (b) finer step 0, fewer steps (secondary; expected to lose)
    _cfg("e9.9_s0.1", "finer_step0", scale=0.1),
    _cfg("e4_s0.1", "finer_step0", extent=4.0, scale=0.1),
    # (c) MAX_CONTAINING_SIZE / neighbor_degree (secondary)
    _cfg("mcs8", "max_containing_size", mcs=8),
    _cfg("mcs10", "max_containing_size", mcs=10),
    _cfg("nd0", "neighbor_degree", neighbor_degree=0),
    _cfg("nd2", "neighbor_degree", neighbor_degree=2),
    # precision trade (correctness trade, never the headline)
    _cfg("p0.002", "precision", precision=0.002),
    _cfg("p0.005", "precision", precision=0.005),
]
REFERENCE = _cfg("reference", "reference", extent=12.0, scale=0.05, precision=1.0e-4, mcs=60)
REFERENCE_B = _cfg("reference_b", "reference", extent=11.0, scale=0.07, precision=1.0e-4, mcs=60)

if _args.list_configs:
    for c in CONFIGS:
        print(c)
    sys.exit(0)
if _args.configs:
    wanted = [c.strip() for c in _args.configs.split(",") if c.strip()]
    unknown = [w for w in wanted if w not in {c["name"] for c in CONFIGS}]
    if unknown:
        raise SystemExit(f"--configs: unknown {unknown}; see --list-configs")
    CONFIGS = [c for c in CONFIGS if c["name"] == "control" or c["name"] in wanted]
CONFIG_NAMES = [c["name"] for c in CONFIGS]

_GRID = al.Grid2D.uniform(shape_native=GRID_SHAPE, pixel_scales=GRID_PIXEL_SCALE)
_GRID_LIMITS = (
    float(np.asarray(_GRID[:, 0]).min()),
    float(np.asarray(_GRID[:, 0]).max()),
    float(np.asarray(_GRID[:, 1]).min()),
    float(np.asarray(_GRID[:, 1]).max()),
)


def make_solver(cfg):
    common = dict(
        pixel_scale_precision=cfg["precision"],
        magnification_threshold=0.1,
        neighbor_degree=cfg["neighbor_degree"],
    )
    if cfg["extent"] is None and cfg["scale"] == GRID_PIXEL_SCALE:
        # The production construction, byte for byte (image_plane.py).
        return al.PointSolver.for_grid(grid=_GRID, **common)
    if cfg["extent"] is None:
        y_min, y_max, x_min, x_max = _GRID_LIMITS
    else:
        e = float(cfg["extent"])
        y_min, y_max, x_min, x_max = -e, e, -e, e
    return al.PointSolver.for_limits_and_scale(
        y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max, scale=cfg["scale"], **common
    )


def geometry(cfg) -> tuple:
    s = make_solver(cfg)
    return (float(s.y_min), float(s.y_max), float(s.x_min), float(s.x_max), float(s.scale))


def static_description(cfg) -> dict:
    s = make_solver(cfg)
    g = geometry(cfg)
    n_triangles = int(_coordinate_array._lattice_coordinates(*g).shape[0])
    vertices, _ = _coordinate_array.static_vertex_table(*g)
    n_steps = int(s.n_steps)
    return {
        "extent_half_width": [g[1], g[3]],
        "limits": list(g[:4]),
        "scale": g[4],
        "pixel_scale_precision": cfg["precision"],
        "n_steps": n_steps,
        "n_steps_formula": f"ceil(log2({g[4]} / {cfg['precision']}))",
        "last_step_triangle_side": g[4] / 2 ** (n_steps - 1),
        "step0_triangles": n_triangles,
        "step0_rows": int(vertices.shape[0]),
        "max_containing_size": cfg["max_containing_size"],
        "neighbor_degree": cfg["neighbor_degree"],
    }


# ---------------------------------------------------------------------------
# MAX_CONTAINING_SIZE patch + trace-time recorder
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def max_containing_size(mcs: int):
    """Set ``MAX_CONTAINING_SIZE`` for the duration of tracing.

    The module constant is bound as a DEFAULT ARGUMENT of ``ArrayTriangles.__init__``
    and ``ArrayTriangles.for_limits_and_scale`` at import time, so re-binding the
    constant alone changes nothing: the defaults tuples are swapped too, and all
    three are restored by value on exit. Proof it took is taken per route from the
    traced solution shape (``(mcs, 2)``).
    """
    init = _triangles_array.ArrayTriangles.__init__
    fls = _triangles_array.ArrayTriangles.for_limits_and_scale.__func__
    saved = (init.__defaults__, fls.__defaults__, _triangles_array.MAX_CONTAINING_SIZE)
    init.__defaults__ = (int(mcs),)
    fls.__defaults__ = (int(mcs),)
    _triangles_array.MAX_CONTAINING_SIZE = int(mcs)
    try:
        yield
    finally:
        init.__defaults__, fls.__defaults__, _triangles_array.MAX_CONTAINING_SIZE = saved


@contextlib.contextmanager
def trace_recorder():
    """Record the ray-traced row count of every solver step while tracing."""
    original = _shape_solver.AbstractSolver._plane_triangles
    record = {"rows": []}

    def recording(self, tracer, triangles, xp, plane_redshift):
        record["rows"].append(int(np.shape(triangles.vertices)[0]))
        return original(
            self, tracer=tracer, triangles=triangles, xp=xp, plane_redshift=plane_redshift
        )

    _shape_solver.AbstractSolver._plane_triangles = recording
    try:
        yield record
    finally:
        _shape_solver.AbstractSolver._plane_triangles = original
    if _shape_solver.AbstractSolver._plane_triangles is not original:
        raise RuntimeError("trace recorder failed to restore AbstractSolver._plane_triangles")


# ---------------------------------------------------------------------------
# Helpers lifted from static_lattice_ab.py (phase 3) -- local copies
# ---------------------------------------------------------------------------


def _rss_bytes():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _memory_analysis(compiled):
    try:
        stats = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"
    if stats is None:
        return "unavailable: None"
    out = {}
    for name in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_temp_size_in_bytes",
    ):
        value = getattr(stats, name, None)
        if value is not None:
            out[name] = int(value)
    return out


def _flops(compiled):
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list | tuple):
            cost = cost[0] if cost else {}
        value = (cost or {}).get("flops")
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"


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


def _timed_call(executable, argument):
    t0 = time.perf_counter()
    out = block(executable(argument))
    return time.perf_counter() - t0, out


def _vector_stream(model, n: int, seed: int) -> np.ndarray:
    """Fixed-seed physical vectors: entry 0 the prior medians, the rest U(0.25,0.75) draws."""
    rng = np.random.default_rng(seed)
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for _ in range(n - 1):
        unit = rng.uniform(0.25, 0.75, size=model.prior_count)
        vectors.append(np.asarray(model.vector_from_unit_vector(unit), dtype=float))
    return np.stack(vectors)


def _prior_draws(model, n: int, seed: int) -> np.ndarray:
    """Fixed-seed draws of the FULL prior (unit cube U(0,1), not the timing stream's core)."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            np.asarray(
                model.vector_from_unit_vector(rng.uniform(1e-4, 1 - 1e-4, size=model.prior_count)),
                dtype=float,
            )
            for _ in range(n)
        ]
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
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


# ---------------------------------------------------------------------------
# Models, dataset, closures
# ---------------------------------------------------------------------------

dataset_path = _ROOT / "dataset" / "point_source" / "simple"
auto_simulate_if_missing(
    dataset_path, dataset_type="point_source", instrument="simple", workspace_root=_ROOT
)
DATASET = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")
POSITION_SIGMA = float(np.asarray(DATASET.positions_noise_map)[0])


def _mass_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    return mass


def _solved_model():
    lens = af.Model(al.Galaxy, redshift=0.5, mass=_mass_model())
    source = af.Model(al.Galaxy, redshift=1.0, point_0=af.Model(al.ps.PointSolved))
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _stress_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
    mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
    mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=2.0)
    mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
    mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
    point = af.Model(al.ps.PointFlux)
    point.centre.centre_0 = af.UniformPrior(lower_limit=-0.4, upper_limit=0.4)
    point.centre.centre_1 = af.UniformPrior(lower_limit=-0.4, upper_limit=0.4)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


MODEL_SOLVED = _solved_model()
MODEL_STRESS = _stress_model()
# Load-bearing for the gradient gate: without it jax.grad of this likelihood is
# identically ZERO (forward values unaffected).
register_model_pytrees(MODEL_SOLVED)
register_model_pytrees(MODEL_STRESS)


def _analysis(cfg, fit_cls):
    return al.AnalysisPoint(
        dataset=DATASET, solver=make_solver(cfg), fit_positions_cls=fit_cls, use_jax=True
    )


def loglike_factory(cfg):
    def factory():
        analysis = _analysis(cfg, al.FitPositionsImagePairAllSolved)

        def log_likelihood(vector):
            instance = MODEL_SOLVED.instance_from_vector(vector=vector, xp=jnp)
            return analysis.log_likelihood_function(instance=instance)

        return log_likelihood

    return factory


def solved_positions_factory(cfg):
    """vector -> (log L, padded model positions) on the solved likelihood."""

    def factory():
        analysis = _analysis(cfg, al.FitPositionsImagePairAllSolved)

        def fn(vector):
            instance = MODEL_SOLVED.instance_from_vector(vector=vector, xp=jnp)
            fit = analysis.fit_from(instance=instance)
            md = fit.positions.model_data
            return fit.log_likelihood, jnp.asarray(getattr(md, "array", md))

        return fn

    return factory


def stress_positions_factory(cfg):
    """vector -> padded model positions for the given lens + source centre."""

    def factory():
        analysis = _analysis(cfg, al.FitPositionsImagePairAll)

        def fn(vector):
            instance = MODEL_STRESS.instance_from_vector(vector=vector, xp=jnp)
            md = analysis.fit_from(instance=instance).positions.model_data
            return jnp.asarray(getattr(md, "array", md))

        return fn

    return factory


def compile_route(
    cfg, factory, example, *, label, expect_positions_rows=False, require_solver=True
):
    """Lower a FRESH closure under cfg's MAX_CONTAINING_SIZE after jax.clear_caches()."""
    jax.clear_caches()
    fn = factory()
    rss0 = _rss_bytes()
    with max_containing_size(cfg["max_containing_size"]), trace_recorder() as record:
        t0 = time.perf_counter()
        lowered = jax.jit(fn).lower(example)
        lower_s = time.perf_counter() - t0
    out_shapes = [tuple(s.shape) for s in jax.tree_util.tree_leaves(lowered.out_info)]
    if require_solver and not record["rows"]:
        raise RuntimeError(f"FAIL [{label}]: the trace recorder saw no solver step")
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    rss1 = _rss_bytes()
    t0 = time.perf_counter()
    block(compiled(example))
    first_call_s = time.perf_counter() - t0
    mcs_proof = None
    if expect_positions_rows:
        pos_shapes = [s for s in out_shapes if len(s) >= 2 and s[-1] == 2]
        mcs_proof = {
            "positions_shape": list(pos_shapes[-1]) if pos_shapes else None,
            "expected_rows": cfg["max_containing_size"],
        }
        mcs_proof["pass"] = bool(pos_shapes and pos_shapes[-1][-2] == cfg["max_containing_size"])
        if not mcs_proof["pass"]:
            raise RuntimeError(
                f"FAIL [{label}]: MAX_CONTAINING_SIZE patch did not take: {mcs_proof}"
            )
    return compiled, {
        "lower_s": float(lower_s),
        "compile_s": float(compile_s),
        "first_call_s": float(first_call_s),
        "traced_rows_per_step": record["rows"],
        "output_shapes": [list(s) for s in out_shapes],
        "flops": _flops(compiled),
        "memory_analysis": _memory_analysis(compiled),
        "rss_delta_bytes": (rss1 - rss0) if rss0 is not None and rss1 is not None else None,
        "max_containing_size_proof": mcs_proof,
    }


# ---------------------------------------------------------------------------
# Image-set comparison
# ---------------------------------------------------------------------------


def finite_positions(p):
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    return p[np.all(np.isfinite(p), axis=1)]


def distinct(p, tol=DISTINCT_TOL):
    """Greedy merge of positions within tol (duplicates of one image from tie triangles)."""
    out = []
    for q in p:
        if not any(np.hypot(*(q - r)) < tol for r in out):
            out.append(q)
    return np.asarray(out).reshape(-1, 2)


def compare_sets(ref, cfg_pos):
    """Multiplicity + nearest-neighbour errors of cfg's distinct images vs the reference's."""
    r = distinct(finite_positions(ref))
    c = distinct(finite_positions(cfg_pos))
    out = {
        "ref_raw": int(finite_positions(ref).shape[0]),
        "cfg_raw": int(finite_positions(cfg_pos).shape[0]),
        "ref_distinct": int(r.shape[0]),
        "cfg_distinct": int(c.shape[0]),
    }
    out["multiplicity_match"] = out["ref_distinct"] == out["cfg_distinct"]
    errors = []
    if r.size and c.size:
        d = np.hypot(r[:, None, 0] - c[None, :, 0], r[:, None, 1] - c[None, :, 1])
        errors = list(d.min(axis=1))  # every reference image -> nearest cfg image
        extra = list(d.min(axis=0))  # every cfg image -> nearest reference image
        out["hausdorff"] = float(max(max(errors), max(extra)))
    else:
        out["hausdorff"] = None if (r.size or c.size) else 0.0
    out["errors"] = [float(e) for e in errors]
    return out


def evaluate_all(compiled, draws, *, returns_logl):
    logls, positions = [], []
    for v in draws:
        out = block(compiled(jnp.asarray(v)))
        if returns_logl:
            logls.append(float(out[0]))
            positions.append(np.asarray(out[1], dtype=float))
        else:
            positions.append(np.asarray(out, dtype=float))
    return logls, positions


def completeness_summary(
    name, comparisons, logl=None, logl_ref=None, logl_control=None, seeds=None
):
    n = len(comparisons)
    mult = [c["multiplicity_match"] for c in comparisons]
    raw = [c["cfg_raw"] == c["ref_raw"] for c in comparisons]
    errs = np.asarray([e for c in comparisons if c["multiplicity_match"] for e in c["errors"]])
    mismatches = [
        {
            "draw": i,
            "seed_draw_index": (seeds[i] if seeds else i),
            "ref_distinct": c["ref_distinct"],
            "cfg_distinct": c["cfg_distinct"],
            "ref_raw": c["ref_raw"],
            "cfg_raw": c["cfg_raw"],
        }
        for i, c in enumerate(comparisons)
        if not c["multiplicity_match"]
    ]
    out = {
        "n_draws": n,
        "multiplicity_agreement": float(np.mean(mult)) if n else None,
        "raw_count_agreement": float(np.mean(raw)) if n else None,
        "n_multiplicity_mismatch": int(n - sum(mult)),
        "mismatch_examples": mismatches[:10],
        "max_position_error": float(errs.max()) if errs.size else None,
        "p99_position_error": float(np.percentile(errs, 99)) if errs.size else None,
        "median_position_error": float(np.median(errs)) if errs.size else None,
        "ref_distinct_histogram": {
            str(k): int(v)
            for k, v in zip(
                *np.unique([c["ref_distinct"] for c in comparisons], return_counts=True)
            )
        },
    }
    if logl is not None:
        a = np.asarray(logl)
        for key, ref in (("vs_reference", logl_ref), ("vs_control", logl_control)):
            if ref is None:
                continue
            b = np.asarray(ref)
            both = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a - b)[both]
            out[f"abs_delta_logl_{key}"] = {
                "max": float(d.max()) if d.size else None,
                "p99": float(np.percentile(d, 99)) if d.size else None,
                "median": float(np.median(d)) if d.size else None,
                "n_nonfinite_mismatch": int(np.sum(np.isfinite(a) != np.isfinite(b))),
            }
    return out


# ===========================================================================
# Run
# ===========================================================================

progress = []


def note(msg):
    line = f"[{time.perf_counter() - _WALL_START:7.1f} s] {msg}"
    print(line, flush=True)
    progress.append(line)


descriptions = {c["name"]: static_description(c) for c in CONFIGS + [REFERENCE, REFERENCE_B]}
note(
    "configs: "
    + ", ".join(f"{n}(n={d['n_steps']},rows={d['step0_rows']})" for n, d in descriptions.items())
)

stream = _vector_stream(MODEL_SOLVED, N_STREAM, SEED)
prior_draws = _prior_draws(MODEL_SOLVED, N_DRAWS, SEED + 1)
stress_draws = _prior_draws(MODEL_STRESS, N_STRESS, SEED + 2) if N_STRESS else np.zeros((0, 1))

# --- 1. completeness: references ------------------------------------------------
completeness = {"prior": {}, "stress": {}}
raw_positions = {"prior": {}, "stress": {}}
route_compile = {}
ref_logl = {}
for ref_cfg in (REFERENCE, REFERENCE_B):
    compiled, rec = compile_route(
        ref_cfg,
        solved_positions_factory(ref_cfg),
        jnp.asarray(prior_draws[0]),
        label=f"{ref_cfg['name']}_prior",
        expect_positions_rows=True,
    )
    route_compile[f"{ref_cfg['name']}_prior"] = rec
    logl, pos = evaluate_all(compiled, prior_draws, returns_logl=True)
    ref_logl[ref_cfg["name"]] = logl
    raw_positions["prior"][ref_cfg["name"]] = pos
    if N_STRESS:
        compiled, rec = compile_route(
            ref_cfg,
            stress_positions_factory(ref_cfg),
            jnp.asarray(stress_draws[0]),
            label=f"{ref_cfg['name']}_stress",
            expect_positions_rows=True,
        )
        route_compile[f"{ref_cfg['name']}_stress"] = rec
        _, pos = evaluate_all(compiled, stress_draws, returns_logl=False)
        raw_positions["stress"][ref_cfg["name"]] = pos
    note(
        f"reference {ref_cfg['name']} solved on {N_DRAWS} prior + {N_STRESS} stress draws "
        f"(compile {rec['compile_s']:.1f} s)"
    )

ref_max_kept = {
    s: int(max(finite_positions(p).shape[0] for p in raw_positions[s]["reference"]))
    for s in raw_positions
    if raw_positions[s].get("reference")
}
reference_floor = {}
for s in raw_positions:
    if not raw_positions[s].get("reference"):
        continue
    comps = [
        compare_sets(a, b)
        for a, b in zip(raw_positions[s]["reference"], raw_positions[s]["reference_b"])
    ]
    reference_floor[s] = completeness_summary(
        "reference_b",
        comps,
        logl=ref_logl["reference_b"] if s == "prior" else None,
        logl_ref=ref_logl["reference"] if s == "prior" else None,
    )
note(
    f"reference floor: {json.dumps({s: {k: v for k, v in f.items() if 'example' not in k} for s, f in reference_floor.items()})}"
)

# --- 2. completeness: every config ---------------------------------------------
config_logl = {}
for cfg in CONFIGS:
    compiled, rec = compile_route(
        cfg,
        solved_positions_factory(cfg),
        jnp.asarray(prior_draws[0]),
        label=f"{cfg['name']}_prior",
        expect_positions_rows=True,
    )
    route_compile[f"{cfg['name']}_prior"] = rec
    logl, pos = evaluate_all(compiled, prior_draws, returns_logl=True)
    config_logl[cfg["name"]] = logl
    raw_positions["prior"][cfg["name"]] = pos
    if N_STRESS:
        compiled, rec = compile_route(
            cfg,
            stress_positions_factory(cfg),
            jnp.asarray(stress_draws[0]),
            label=f"{cfg['name']}_stress",
            expect_positions_rows=True,
        )
        route_compile[f"{cfg['name']}_stress"] = rec
        _, pos = evaluate_all(compiled, stress_draws, returns_logl=False)
        raw_positions["stress"][cfg["name"]] = pos

for cfg in CONFIGS:
    name = cfg["name"]
    comps = [
        compare_sets(a, b)
        for a, b in zip(raw_positions["prior"]["reference"], raw_positions["prior"][name])
    ]
    completeness["prior"][name] = completeness_summary(
        name,
        comps,
        logl=config_logl[name],
        logl_ref=ref_logl["reference"],
        logl_control=config_logl["control"],
    )
    if N_STRESS:
        comps = [
            compare_sets(a, b)
            for a, b in zip(raw_positions["stress"]["reference"], raw_positions["stress"][name])
        ]
        completeness["stress"][name] = completeness_summary(name, comps)
    c = completeness["prior"][name]
    s = completeness["stress"].get(name, {})
    note(
        f"completeness {name:<11} prior mult {c['multiplicity_agreement']:.3f} raw "
        f"{c['raw_count_agreement']:.3f} maxerr {c['max_position_error']} "
        f"|dlogL|ref {c['abs_delta_logl_vs_reference']['max']}  stress mult "
        f"{s.get('multiplicity_agreement')} maxerr {s.get('max_position_error')}"
    )


def admissible(name) -> bool:
    c = completeness["prior"][name]
    return bool(
        c["multiplicity_agreement"] == 1.0
        and c["max_position_error"] is not None
        and c["max_position_error"] <= ADMISSIBLE_POSITION_TOL
    )


admissibility = {c["name"]: admissible(c["name"]) for c in CONFIGS}
_control_side = descriptions["control"]["last_step_triangle_side"]
precision_equivalent = {
    c["name"]: bool(
        descriptions[c["name"]]["last_step_triangle_side"] <= _control_side * (1 + 1e-9)
    )
    for c in CONFIGS
}


def _image_extent(pos_list):
    """max |y| and |x| of the reference's distinct images: the extent the draws need."""
    pts = [distinct(finite_positions(p)) for p in pos_list]
    pts = np.concatenate([p for p in pts if p.size] or [np.zeros((0, 2))])
    if not pts.size:
        return None
    r = np.hypot(pts[:, 0], pts[:, 1])
    return {
        "max_abs_y": float(np.abs(pts[:, 0]).max()),
        "max_abs_x": float(np.abs(pts[:, 1]).max()),
        "max_abs_coord": float(np.abs(pts).max()),
        "p99_abs_coord": float(np.percentile(np.abs(pts).max(axis=1), 99)),
        "max_radius": float(r.max()),
    }


reference_image_extent = {
    s: _image_extent(raw_positions[s]["reference"])
    for s in raw_positions
    if raw_positions[s].get("reference")
}
note(f"reference image extent: {reference_image_extent}")

# Default vs reference (headline check).
default_vs_reference = {
    "prior": completeness["prior"]["control"],
    "stress": completeness["stress"].get("control"),
    "agrees_on_every_prior_draw": completeness["prior"]["control"]["n_multiplicity_mismatch"] == 0,
    "agrees_on_every_stress_draw": (
        completeness["stress"]["control"]["n_multiplicity_mismatch"] == 0 if N_STRESS else None
    ),
}
if N_STRESS:
    ex = []
    for m in completeness["stress"]["control"]["mismatch_examples"][:5]:
        i = m["draw"]
        ex.append(
            {
                **m,
                "vector": stress_draws[i].tolist(),
                "vector_paths": [".".join(p) for p in MODEL_STRESS.paths],
                "reference_positions": distinct(
                    finite_positions(raw_positions["stress"]["reference"][i])
                ).tolist(),
                "control_positions": finite_positions(
                    raw_positions["stress"]["control"][i]
                ).tolist(),
            }
        )
    default_vs_reference["stress_mismatch_examples"] = ex

# --- 3. timing: interleaved rounds over every config ---------------------------
note("timing: compiling every route (fresh closure + clear_caches)")
arguments = [jnp.asarray(v) for v in stream]
executables, timing_compile = {}, {}
for cfg in CONFIGS:
    compiled, rec = compile_route(cfg, loglike_factory(cfg), arguments[0], label=cfg["name"])
    executables[cfg["name"]] = compiled
    timing_compile[cfg["name"]] = rec

warm_ms = {}
for name, ex in executables.items():
    warm_ms[name] = [1e3 * _timed_call(ex, arguments[w % len(arguments)])[0] for w in range(N_WARM)]

times = {n: [] for n in CONFIG_NAMES}
per_round = {n: [] for n in CONFIG_NAMES}
values = {n: {} for n in CONFIG_NAMES}
deterministic = {n: True for n in CONFIG_NAMES}
call_index = 0
for r in range(N_ROUNDS):
    order = CONFIG_NAMES[r % len(CONFIG_NAMES) :] + CONFIG_NAMES[: r % len(CONFIG_NAMES)]
    start = call_index
    for name in order:
        rt = []
        for c in range(N_CALLS):
            k = (start + c) % len(arguments)
            dt, out = _timed_call(executables[name], arguments[k])
            rt.append(dt)
            v = float(out)
            if k in values[name]:
                if not (v == values[name][k] or (np.isnan(v) and np.isnan(values[name][k]))):
                    deterministic[name] = False
            else:
                values[name][k] = v
        times[name].extend(rt)
        per_round[name].append(float(np.median(rt)))
    call_index = start + N_CALLS
note(f"timing: {N_ROUNDS} rounds x {N_CALLS} calls x {len(CONFIG_NAMES)} routes done")

control_stats = _stats_ms(times["control"])
rows = {}
for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    stats = _stats_ms(times[name])
    paired = np.asarray(per_round["control"]) / np.asarray(per_round[name])
    rows[name] = {
        "config": cfg,
        "static": descriptions[name],
        "stats": stats,
        "speedup_vs_control": _median_ratio(times["control"], times[name], BOOTSTRAP_SEED + i),
        "paired_round_speedup": {
            "median": float(np.median(paired)),
            "p10": float(np.percentile(paired, 10)),
            "p90": float(np.percentile(paired, 90)),
        },
        "compile": timing_compile[name],
        "warm_ms": warm_ms[name],
        "per_call_ms": [t * 1e3 for t in times[name]],
        "deterministic": deterministic[name],
        "fiducial_log_likelihood": values[name].get(0),
        "stream_log_likelihood": [values[name][k] for k in sorted(values[name])],
        "admissible": admissibility[name],
        "completeness_prior": completeness["prior"][name],
        "completeness_stress": completeness["stress"].get(name),
    }

# --- 4. gates -------------------------------------------------------------------
fid = rows["control"]["fiducial_log_likelihood"]
gates = {
    "fiducial_bit_exact": {
        "value": fid,
        "expected": FIDUCIAL_SOLVED_LOG_L,
        "repr": repr(fid),
        "pass": fid == FIDUCIAL_SOLVED_LOG_L,
    },
    "max_containing_size_patch": {
        n: route_compile[f"{n}_prior"]["max_containing_size_proof"] for n in CONFIG_NAMES
    },
    "patch_restored": {
        "module_constant": int(_triangles_array.MAX_CONTAINING_SIZE),
        "init_default": _triangles_array.ArrayTriangles.__init__.__defaults__,
        "pass": int(_triangles_array.MAX_CONTAINING_SIZE) == PRODUCTION_MCS
        and _triangles_array.ArrayTriangles.__init__.__defaults__ == (PRODUCTION_MCS,),
    },
}
note(
    f"fiducial gate: {repr(fid)} == {FIDUCIAL_SOLVED_LOG_L!r}: {gates['fiducial_bit_exact']['pass']}"
)

grad_gate = {}
for cfg in CONFIGS:
    name = cfg["name"]
    if not (admissibility[name] or name == "control"):
        continue

    def gfactory(cfg=cfg):
        return jax.grad(loglike_factory(cfg)())

    compiled, rec = compile_route(cfg, gfactory, arguments[0], label=f"{name}_grad")
    grads = np.stack([np.asarray(block(compiled(a)), dtype=float) for a in arguments[:N_GRAD]])
    grad_gate[name] = {
        "compile_s": rec["compile_s"],
        "all_finite": bool(np.isfinite(grads).all()),
        "non_zero_per_instance": [bool(np.any(g != 0.0)) for g in grads],
        "gradients": grads.tolist(),
    }
    grad_gate[name]["pass"] = grad_gate[name]["all_finite"] and all(
        grad_gate[name]["non_zero_per_instance"]
    )
    if name != "control":
        grad_gate[name]["max_rel_delta_vs_control"] = None
gctrl = np.asarray(grad_gate["control"]["gradients"])
for name, g in grad_gate.items():
    ga = np.asarray(g["gradients"])
    g["max_rel_delta_vs_control"] = float(
        np.max(np.abs(ga - gctrl) / np.maximum(np.abs(gctrl), 1e-300))
    )
gates["grad"] = grad_gate
note("grad gate: " + ", ".join(f"{n}={g['pass']}" for n, g in grad_gate.items()))

# --- 5. best admissible + vmap batches -----------------------------------------
candidates_any = [
    n
    for n in CONFIG_NAMES
    if n != "control" and admissibility[n] and rows[n]["config"]["block"] != "precision"
]
candidates = [n for n in candidates_any if precision_equivalent[n]]


def _fastest(names):
    return max(names, key=lambda n: rows[n]["speedup_vs_control"]["ratio"]) if names else None


best = _fastest(candidates)
best_any_precision = _fastest(candidates_any)
note(f"best admissible (precision-equivalent): {best}; any precision: {best_any_precision}")

vmap_block = None
if best is not None:
    vroutes = [(n, b) for n in ("control", best) for b in VMAP_BATCHES]
    vexec, vcompile, vargs = {}, {}, {}
    for n, b in vroutes:
        cfg = next(c for c in CONFIGS if c["name"] == n)
        nb = max(1, len(stream) // b)
        batches = [
            jnp.asarray(stream[[(j * b + k) % len(stream) for k in range(b)]]) for j in range(nb)
        ]
        vargs[(n, b)] = batches

        def vfactory(cfg=cfg):
            return jax.vmap(loglike_factory(cfg)())

        vexec[(n, b)], vcompile[(n, b)] = compile_route(
            cfg, vfactory, batches[0], label=f"{n}_vmap{b}"
        )
    for key, ex in vexec.items():
        for w in range(N_WARM):
            _timed_call(ex, vargs[key][w % len(vargs[key])])
    vtimes = {key: [] for key in vroutes}
    vvals = {key: {} for key in vroutes}
    for r in range(N_ROUNDS):
        order = vroutes[r % len(vroutes) :] + vroutes[: r % len(vroutes)]
        for key in order:
            for c in range(N_CALLS):
                k = (r * N_CALLS + c) % len(vargs[key])
                dt, out = _timed_call(vexec[key], vargs[key][k])
                vtimes[key].append(dt)
                vvals[key].setdefault(k, np.asarray(out, dtype=float).tolist())
    vmap_block = {"best": best, "batches": list(VMAP_BATCHES), "rows": {}}
    for i, (n, b) in enumerate(vroutes):
        st = _stats_ms(vtimes[(n, b)])
        vmap_block["rows"][f"{n}_vmap{b}"] = {
            "config": n,
            "batch": b,
            "stats_per_batch": st,
            "median_ms_per_likelihood": st["median_ms"] / b,
            "compile": vcompile[(n, b)],
            "values_first_batch": vvals[(n, b)].get(0),
        }
    for b in VMAP_BATCHES:
        vmap_block["rows"][f"{best}_vmap{b}"]["speedup_vs_control_same_batch"] = _median_ratio(
            vtimes[("control", b)], vtimes[(best, b)], BOOTSTRAP_SEED + 100 + b
        )
    # vmap vs scalar consistency (control, first batch)
    for n in ("control", best):
        for b in VMAP_BATCHES:
            first = vvals[(n, b)].get(0)
            scalar = [values[n].get(k) for k in range(b)]
            if first is not None and all(s is not None for s in scalar):
                d = np.abs(np.asarray(first) - np.asarray(scalar))
                vmap_block["rows"][f"{n}_vmap{b}"]["max_abs_delta_vs_scalar"] = float(np.nanmax(d))
    note(
        "vmap: "
        + ", ".join(
            f"{k} {v['median_ms_per_likelihood']:.3f} ms/L" for k, v in vmap_block["rows"].items()
        )
    )

# --- 6. step-0 ray trace vs containment split per geometry ---------------------
note("step-0 split: ray trace vs ray trace + containment per geometry")


def step0_factory(cfg, stage):
    def factory():
        solver = make_solver(cfg)
        analysis = _analysis(cfg, al.FitPositionsImagePairAllSolved)

        def fn(vector):
            instance = MODEL_SOLVED.instance_from_vector(vector=vector, xp=jnp)
            fit = analysis.fit_from(instance=instance).positions
            shape = Point(*fit.source_plane_coordinate)
            tri = solver._initial_triangles(jnp)
            if stage == "source_centre":
                return jnp.stack(fit.source_plane_coordinate)
            plane = solver._plane_triangles(
                tracer=fit.tracer, triangles=tri, xp=jnp, plane_redshift=fit.plane_redshift
            )
            if stage == "ray_trace":
                # A scalar reduction forces every step-0 deflection but writes no
                # (N, 3, 2) output, so containment is measured against it.
                return jnp.sum(plane.vertices)
            if stage == "ray_trace_materialised":
                # image_plane.py's ray-trace prefix (job 356365): returns the
                # gathered (N, 3, 2) triangles, i.e. includes their materialisation.
                return plane.triangles
            return plane.containing_indices(shape=shape)

        return fn

    return factory


geoms = {}
for cfg in CONFIGS:
    key = (cfg["extent"], cfg["scale"])
    geoms.setdefault(key, cfg)
s0_exec = {}
stages = ("source_centre", "ray_trace", "ray_trace_materialised", "containment")
for key, cfg in geoms.items():
    for stage in stages:
        s0_exec[(key, stage)], _ = compile_route(
            cfg,
            step0_factory(cfg, stage),
            arguments[0],
            label=f"step0_{cfg['name']}_{stage}",
            require_solver=stage != "source_centre",
        )
s0_times = {k: [] for k in s0_exec}
keys = list(s0_exec)
for r in range(N_STEP0_ROUNDS):
    order = keys[r % len(keys) :] + keys[: r % len(keys)]
    for k in order:
        for c in range(N_CALLS):
            s0_times[k].append(
                _timed_call(s0_exec[k], arguments[(r * N_CALLS + c) % len(arguments)])[0]
            )
step0_split = {}
for key, cfg in geoms.items():
    med = {st: float(np.median(s0_times[(key, st)]) * 1e3) for st in stages}
    trace = med["ray_trace"] - med["source_centre"]
    contain = med["containment"] - med["ray_trace"]
    step0_split[cfg["name"]] = {
        "extent": cfg["extent"],
        "scale": cfg["scale"],
        "step0_rows": descriptions[cfg["name"]]["step0_rows"],
        "step0_triangles": descriptions[cfg["name"]]["step0_triangles"],
        "prefix_median_ms": med,
        "ray_trace_ms": trace,
        "containment_ms": contain,
        "triangle_materialisation_ms": med["ray_trace_materialised"] - med["ray_trace"],
        "method": (
            "prefix medians; ray_trace = jnp.sum(plane.vertices) minus source_centre; "
            "containment = containing_indices minus ray_trace (includes the vertices[indices] "
            "gather + Point.mask + jnp.where); ray_trace_materialised = image_plane.py's prefix"
        ),
        "step0_ms": trace + contain,
        "containment_share_of_step0": contain / (trace + contain) if trace + contain > 0 else None,
        "step0_share_of_likelihood": (trace + contain) / rows[cfg["name"]]["stats"]["median_ms"],
        "containment_share_of_likelihood": contain / rows[cfg["name"]]["stats"]["median_ms"],
    }
note(
    "step-0 split: "
    + ", ".join(
        f"{n} trace {v['ray_trace_ms']:.3f} contain {v['containment_ms']:.3f}"
        for n, v in step0_split.items()
    )
)

# --- 7. uncapped containing-triangle counts (NumPy path) -----------------------
# MAX_CONTAINING_SIZE truncates each step's kept set (jnp.where(size=...)). The
# NumPy solver has dynamic shapes and no cap, so it shows how many triangles
# contain beta* at each step before truncation: the margin the production 15
# actually has on the prior draws.
note("uncapped containing counts (NumPy path) on the prior draws")


def containing_counts(cfg, draws):
    solver = make_solver(cfg)
    analysis = al.AnalysisPoint(
        dataset=DATASET,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePairAllSolved,
        use_jax=False,
    )
    per_draw = []
    for v in draws:
        fit = analysis.fit_from(instance=MODEL_SOLVED.instance_from_vector(vector=v)).positions
        shape = Point(*fit.source_plane_coordinate)
        counts = []
        for step in solver.steps(
            tracer=fit.tracer, shape=shape, xp=np, plane_redshift=fit.plane_redshift
        ):
            t = np.asarray(step.filtered_triangles.triangles)
            counts.append(int(np.isfinite(t).all(axis=(1, 2)).sum()))
        per_draw.append(counts)
    arr = np.asarray(per_draw)
    worst = arr.max(axis=1)
    return {
        "n_draws": int(arr.shape[0]),
        "max_per_step": arr.max(axis=0).tolist(),
        "p99_per_step": np.percentile(arr, 99, axis=0).tolist(),
        "median_per_step": np.median(arr, axis=0).tolist(),
        "max_over_steps": int(arr.max()),
        "draws_exceeding_cap": [int(i) for i in np.where(worst > cfg["max_containing_size"])[0]],
        "n_draws_exceeding": {str(k): int((worst > k).sum()) for k in (6, 8, 10, 12, 15)},
        "histogram_max_over_steps": {
            str(k): int(c) for k, c in zip(*np.unique(worst, return_counts=True))
        },
    }


uncapped_counts = {}
for name in dict.fromkeys(n for n in ("control", best, best_any_precision) if n):
    cfg = next(c for c in CONFIGS if c["name"] == name)
    uncapped_counts[name] = containing_counts(cfg, prior_draws)
    u = uncapped_counts[name]
    note(
        f"uncapped {name}: max per step {u['max_per_step']}; draws exceeding cap "
        f"{cfg['max_containing_size']}: {u['draws_exceeding_cap']}"
    )

# ===========================================================================
# Summary
# ===========================================================================

config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_device_fp64"
)
device = device_info_dict()
if device.get("jax_compilation_cache_dir"):
    device["jax_compilation_cache_dir"] = (
        "<scrubbed>/" + Path(device["jax_compilation_cache_dir"]).name
    )

table = []
for name in CONFIG_NAMES:
    r = rows[name]
    cp = r["completeness_prior"]
    cs = r["completeness_stress"] or {}
    table.append(
        {
            "config": name,
            "block": r["config"]["block"],
            "extent": r["static"]["extent_half_width"][0],
            "scale": r["static"]["scale"],
            "precision": r["static"]["pixel_scale_precision"],
            "mcs": r["config"]["max_containing_size"],
            "neighbor_degree": r["config"]["neighbor_degree"],
            "n_steps": r["static"]["n_steps"],
            "step0_rows": r["static"]["step0_rows"],
            "step0_triangles": r["static"]["step0_triangles"],
            "median_ms": r["stats"]["median_ms"],
            "speedup": r["speedup_vs_control"]["ratio"],
            "speedup_ci90": [
                r["speedup_vs_control"]["ci90_low"],
                r["speedup_vs_control"]["ci90_high"],
            ],
            "admissible": r["admissible"],
            "precision_equivalent": precision_equivalent[name],
            "last_step_triangle_side": r["static"]["last_step_triangle_side"],
            "prior_multiplicity_agreement": cp["multiplicity_agreement"],
            "prior_raw_count_agreement": cp["raw_count_agreement"],
            "prior_max_err": cp["max_position_error"],
            "prior_p99_err": cp["p99_position_error"],
            "prior_abs_dlogl_ref_max": cp["abs_delta_logl_vs_reference"]["max"],
            "prior_abs_dlogl_control_max": cp["abs_delta_logl_vs_control"]["max"],
            "stress_multiplicity_agreement": cs.get("multiplicity_agreement"),
            "stress_max_err": cs.get("max_position_error"),
            "compile_s": r["compile"]["compile_s"],
            "flops": r["compile"]["flops"],
            "temp_bytes": (r["compile"]["memory_analysis"] or {}).get("temp_size_in_bytes")
            if isinstance(r["compile"]["memory_analysis"], dict)
            else None,
        }
    )

all_gates_pass = bool(
    gates["fiducial_bit_exact"]["pass"]
    and gates["patch_restored"]["pass"]
    and all(g["pass"] for g in gates["max_containing_size_patch"].values() if g)
    and all(g["pass"] for g in grad_gate.values())
)

summary = {
    "cell": "solver_config_sweep",
    "issue": "PyAutoLabs/autolens_profiling#314",
    "phase": "4a",
    "config_name": config_name,
    "quotable": not config_name.startswith(("laptop", "local")),
    "host_note": (
        "laptop witness: WSL2, interactive load, NOT quotable"
        if config_name.startswith(("laptop", "local"))
        else None
    ),
    "quick": QUICK,
    "library_versions": {
        "autolens": al.__version__,
        "autoarray": aa.__version__,
        "autofit": af.__version__,
    },
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    "source_revisions": source_revisions(_ROOT),
    "autoarray_imported_from": str(Path(aa.__file__).resolve().parent.parent),
    "autolens_imported_from": str(Path(al.__file__).resolve().parent.parent),
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
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "n_stream": N_STREAM,
        "seed": SEED,
        "n_prior_draws": N_DRAWS,
        "n_stress_draws": N_STRESS,
        "step0_rounds": N_STEP0_ROUNDS,
        "vmap_batches": list(VMAP_BATCHES),
        "route_order": "round-robin over every config, start rotated each round",
        "cache_trap_guard": "fresh closure + fresh solver/analysis per route; jax.clear_caches()",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": "90%"},
        "admissible": (
            "100% distinct-multiplicity agreement with the reference on the prior draws AND "
            f'max position error <= {ADMISSIBLE_POSITION_TOL}"'
        ),
        "admissible_position_tol": ADMISSIBLE_POSITION_TOL,
        "position_noise_sigma": POSITION_SIGMA,
        "distinct_tol": DISTINCT_TOL,
        "prior_draws": "full prior: vector_from_unit_vector(U(1e-4, 1-1e-4)); source = solved beta*",
        "stress_draws": (
            "OUTSIDE the prior: Isothermal centre U(-0.05,0.05), theta_E U(1,2), ell_comps "
            "U(-0.2,0.2), source centre U(-0.4,0.4); plain FitPositionsImagePairAll positions"
        ),
        "reference": REFERENCE,
        "reference_b": REFERENCE_B,
        "max_containing_size_patch": (
            "ArrayTriangles.__init__ / for_limits_and_scale __defaults__ + module constant, "
            "tracing only, restored by value"
        ),
        "pytrees": "autofit.jax.register_model(model) for both models before any trace",
    },
    "configs": CONFIGS,
    "static": descriptions,
    "reference_floor": reference_floor,
    "reference_max_finite_positions": ref_max_kept,
    "reference_compile": {k: v for k, v in route_compile.items() if k.startswith("reference")},
    "default_vs_reference": default_vs_reference,
    "admissibility": admissibility,
    "precision_equivalent": precision_equivalent,
    "reference_image_extent": reference_image_extent,
    "extent_working": (
        "prior: theta_E <= 1.6 + 4*0.05 = 1.8; |e| <= 0.0526 + 4*0.01 ~ 0.09 -> q ~ 0.83; SIE "
        "images lie within ~theta_E/sqrt(q) + |beta| ~ 1.98 + 0.1 ~ 2.1 arcsec of the centre, so "
        "boxes of +-6 / +-4 / +-3 / +-2.5 bracket it; reference_image_extent is the measured value"
    ),
    "best_admissible": best,
    "best_admissible_any_precision": best_any_precision,
    "table": table,
    "rows": rows,
    "vmap": vmap_block,
    "step0_split": step0_split,
    "uncapped_containing_counts": uncapped_counts,
    "gates": gates,
    "all_gates_pass": all_gates_pass,
    "completeness_compile": {
        k: v for k, v in route_compile.items() if not k.startswith("reference")
    },
    "progress": progress,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_image",
    default_basename="solver_config_sweep_local",
    cell="solver_config_sweep",
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2, default=str))

# ---------------------------------------------------------------------------
# PNG: speed-up with CI (colour = admissible), and speed-up vs max position error
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 7), constrained_layout=True)
ys = np.arange(len(table))
for y, t in zip(ys, table):
    colour = "#4C72B0" if t["admissible"] else "#C44E52"
    ax1.barh(y, t["speedup"], color=colour, alpha=0.75)
    ax1.plot(t["speedup_ci90"], [y, y], color="black", lw=1)
ax1.axvline(1.0, color="black", lw=0.8)
ax1.set_yticks(ys)
ax1.set_yticklabels(
    [f"{t['config']} (n={t['n_steps']}, rows={t['step0_rows']})" for t in table], fontsize=7
)
ax1.invert_yaxis()
ax1.set_xlabel("speed-up vs control (median ratio, 90% CI); blue = admissible, red = not")
for t in table:
    err = t["prior_max_err"] if t["prior_max_err"] is not None else np.nan
    ax2.scatter(err, t["speedup"], color="#4C72B0" if t["admissible"] else "#C44E52")
    ax2.annotate(t["config"], (err, t["speedup"]), fontsize=6)
ax2.axvline(ADMISSIBLE_POSITION_TOL, color="grey", ls="--", lw=0.8)
ax2.set_xscale("log")
ax2.set_xlabel("max position error vs reference on prior draws [arcsec]")
ax2.set_ylabel("speed-up vs control")
fig.suptitle(
    f"PointSolver config sweep ({config_name}); control {control_stats['median_ms']:.3f} ms; "
    f"best admissible={best}; all_gates_pass={all_gates_pass}",
    fontsize=10,
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 150)
print(
    f"POINTSOLVER CONFIG SWEEP ({config_name}, rounds={N_ROUNDS}, calls={N_CALLS}, "
    f"prior draws={N_DRAWS}, stress draws={N_STRESS})"
)
print("=" * 150)
print(
    f"  {'config':<11} {'block':<20} {'ext':>5} {'scale':>5} {'prec':>6} {'mcs':>3} {'nd':>2} "
    f"{'n':>2} {'rows':>6} {'med_ms':>7} {'speedup [90% CI]':>22} {'adm':>4}p {'mult':>5} "
    f"{'raw':>5} {'maxerr':>9} {'p99err':>9} {'dlogL_ref':>9} {'s_mult':>6} {'s_maxerr':>9}"
)
for t in table:

    def f(x, fmt):
        return format(x, fmt) if isinstance(x, int | float) else "  n/a"

    print(
        f"  {t['config']:<11} {t['block']:<20} {t['extent']:5.2f} {t['scale']:5.2f} "
        f"{t['precision']:6.4f} {t['mcs']:3d} {t['neighbor_degree']:2d} {t['n_steps']:2d} "
        f"{t['step0_rows']:6d} {t['median_ms']:7.3f} {t['speedup']:6.3f} "
        f"[{t['speedup_ci90'][0]:.3f}, {t['speedup_ci90'][1]:.3f}] {str(t['admissible'])[0]:>4}{str(t['precision_equivalent'])[0]} "
        f"{f(t['prior_multiplicity_agreement'], '5.3f')} {f(t['prior_raw_count_agreement'], '5.3f')} "
        f"{f(t['prior_max_err'], '9.2e')} {f(t['prior_p99_err'], '9.2e')} "
        f"{f(t['prior_abs_dlogl_ref_max'], '9.2e')} {f(t['stress_multiplicity_agreement'], '6.3f')} "
        f"{f(t['stress_max_err'], '9.2e')}"
    )
print("-" * 150)
print(
    f"  default agrees with reference on every prior draw: {default_vs_reference['agrees_on_every_prior_draw']}"
)
print(
    f"  default agrees with reference on every stress draw: {default_vs_reference['agrees_on_every_stress_draw']}"
)
print(
    f"  reference floor: {json.dumps({s: {k: v for k, v in fl.items() if 'example' not in k} for s, fl in reference_floor.items()})}"
)
print(f"  best admissible (precision-equivalent): {best}; any precision: {best_any_precision}")
print(f"  reference image extent: {reference_image_extent}")
print(f"  fiducial gate: {gates['fiducial_bit_exact']}")
print(f"  all_gates_pass: {all_gates_pass}")
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
