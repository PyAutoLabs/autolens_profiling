"""
JAX Profiling: PointSolver Static Step-0 Lattice A/B (point-source CPU campaign, phase 3)
========================================================================================

An interleaved, in-process A/B of the JAX ``PointSolver``'s step-0 triangle tiling
inside the production point-source likelihoods (PyAutoArray #568, phase 3).

The mechanism
-------------

``AbstractSolver.steps`` builds the step-0 tiling with
``AbstractSolver._initial_triangles`` and ray-traces ``triangles.vertices``. After
phase 2 (PyAutoArray ``681938ae``) the JAX ``vertices`` table is the flat ``(3N, 2)``
per-triangle table: for the simple solver's +-9.9" / 0.2" lattice that is 69 849
rows, of which only 28 665 are exact-float distinct and only **11 859** are
geometrically distinct (a lattice point shared by up to six triangles is computed
from each triangle's centre and the copies differ by ~1 ulp). The lattice depends
only on the solver geometry (static pytree aux data), so the unique table is a
compile-time constant: deflecting it and gathering back cuts the step-0 deflections
5.9x. After phase 2 deflections are 70 % of the simple call's FLOPs, and step 0 is
about half of them.

The four routes
---------------

Each route swaps ``AbstractSolver._initial_triangles`` for the duration of
**tracing** only (:func:`route_injected`: signature check, call counters, restore by
identity):

- ``control`` -- the flat ``(3N, 2)`` table: ``CoordinateArrayTriangles
  .for_limits_and_scale`` with no table, i.e. the phase-2 main path (69 849 rows
  for the simple lattice).
- ``exact`` -- a table of the exact-float distinct vertex rows (28 665) and the
  inverse index map, built by this cell in NumPy.
- ``lattice`` -- the geometric table keyed on integer lattice positions (11 859),
  built by this cell in NumPy with the same arithmetic as the library's
  ``static_vertex_table``.
- ``library`` -- no substitution: whatever the imported PyAutoLens / PyAutoArray do.
  It **self-labels** from the row count of the grid its step 0 deflects while
  tracing: ``library_matches = control | exact | lattice | neither``. On the phase-3
  branch it must read ``lattice``; on main, ``control``.

``exact`` and ``lattice`` are carried as a subclass of the library's
``CoordinateArrayTriangles`` whose ``_vertices_and_indices`` returns the cell's table,
so both routes run on PyAutoArray main as well as on the branch. Derived lattices
(``for_indexes`` / ``up_sample`` / ``neighborhood``) are built as the base class, so
steps >= 1 are identical in every route.

Every route records the row count of its step-0 deflection grid (from a wrapper of
``AbstractSolver._plane_grid`` armed by the injected ``_initial_triangles``); a route
whose count is not its expected one, or whose substitute never ran while tracing, is a
FAIL, not a number.

The measurement trap
--------------------

``jax`` caches traced jaxprs on function identity. Every route lowers a **fresh**
closure from a factory (fresh ``PointSolver`` and ``AnalysisPoint`` too) after
``jax.clear_caches()``. Once compiled an executable no longer depends on the patch,
so the four executables coexist and the timed calls run with nothing patched.

The A/B protocol
----------------

Rows (``--models``): ``simple`` = the phase-1 harness model, dataset and solver (SIE,
``PointSolver.for_grid`` 100x100 @ 0.2", precision 0.001") with the fused
``FitPositionsImagePairAllSolved`` (``solved``) and ``FitPositionsImagePairAll``
(``plain``) likelihoods, plus a ``vmap`` batch row of the solved likelihood;
``cluster`` = the two-source 13-component model (200x200 @ 0.7", precision 0.01")
with ``FitPositionsImagePairRepeat`` / ``...Solved`` summed over both systems.

Each compiled likelihood takes the physical parameter vector and builds the instance
inside the trace. Per row: compile each route once, warm >= 3 calls, then
``--rounds`` (default 20) round-robin rounds, route order rotating, ``--calls``
(default 20) calls per route per round. Per route: median / p10 / p90 / min / max
per-call ms; bootstrap 90 % intervals of the median ratios ``control / lattice``,
``control / exact``, ``exact / lattice`` and ``library / lattice``; the minimum
detectable improvement ``(p90 - p10) / median`` of the control route.

Per route and row the cell also records: lower / compile / first-call seconds;
``cost_analysis`` FLOPs; ``compiled.memory_analysis()`` (argument / output / temp /
generated-code bytes); the process RSS before and after the lower + compile; the
HLO line count and hash; and, once per run, the cell table build time (cold and
cache hit), the table sizes in MB, the library ``static_vertex_table`` build and
cache-hit time when it exists, and the XLA CPU thread count observed after the
first compile (threads of this process), the scheduler affinity and ``os.cpu_count``.

Correctness gates (the tolerance gate the human chose, 2026-09-24; recorded, never
asserted): log likelihood relative difference <= 1e-12 vs control on every streamed
instance; solved image positions: identical finite-image counts per instance and max
|delta| <= 1e-10; for the simple solved likelihood ``jax.grad`` finite, NON-zero and
``allclose(rtol=1e-10, atol=1e-12)``; ``vmap`` values equal to the per-instance scalar
values (rtol 1e-12); for simple plain, the step-0 ``containing_indices`` set of each
gate instance is identical across routes.

Constant folding
----------------

Every PyAuto submit runs with ``--xla_disable_hlo_passes=constant_folding`` (set by
PyAutoNerves ``jax_wrapper`` whenever the substring is missing). With a constant
table, folding could pre-evaluate the control route's lattice arithmetic too, so the
comparison is re-run with folding ENABLED in a separate process: ``--constant-folding``
replaces the flag, before ``jax`` is imported, with the dummy pass name
``--xla_disable_hlo_passes=constant_folding_probe_disabled`` -- it contains the
substring, so Nerves does not re-add the real flag, and names no real pass, so XLA
folds. The JSON's ``constant_folding.probe`` compiles a dot of two 1000-element
constants and records whether the optimised HLO entry still carries a computation
(folding off) or only a literal (folding on): the probe, not the flag string, is the evidence, and
the cell refuses to run when the probe disagrees with the request. (Elementwise
arithmetic on constants is folded in both modes, so it is no probe.)

Output
------

``results/breakdown/point_source_image/static_lattice_ab_<config_name>.{json,png}``
(``..._constant_folding`` suffix for the folding run). As in the phase-2 cell the JSON
carries no top-level ``autolens_version``, so ``build_readme.py`` does not auto-table
it. A ``laptop_*`` config is load-inflated and is recorded ``quotable: false``.
"""

import os
import sys
import time
from pathlib import Path

# --constant-folding must act BEFORE jax (or autonerves) is imported.
CONSTANT_FOLDING = "--constant-folding" in sys.argv
_FOLDING_DISABLE = "--xla_disable_hlo_passes=constant_folding"
_FOLDING_DUMMY = "--xla_disable_hlo_passes=constant_folding_probe_disabled"
_XLA_FLAGS_REQUESTED = os.environ.get("XLA_FLAGS")
if CONSTANT_FOLDING:
    _flags = [f for f in (_XLA_FLAGS_REQUESTED or "").split() if f != _FOLDING_DISABLE]
    os.environ["XLA_FLAGS"] = " ".join(_flags + [_FOLDING_DUMMY])


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
import functools
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
    "--constant-folding",
    action="store_true",
    help="re-enable XLA constant folding (acted on before jax is imported; see docstring)",
)
_cell_parser.add_argument(
    "--quick", action="store_true", help="rounds=5, calls=3, simple model only"
)
_args = _cli.parse_cell_args(_cell_parser)

QUICK = bool(_args.quick)
N_ROUNDS = _args.rounds if _args.rounds is not None else (5 if QUICK else 20)
N_CALLS = _args.calls if _args.calls is not None else (3 if QUICK else 20)
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
from autoarray.structures.triangles.abstract import HEIGHT_FACTOR  # noqa: E402
from autoarray.structures.triangles.shape import Point  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402
from autolens.point.solver import shape_solver as _shape_solver  # noqa: E402

ROUTES = ("control", "exact", "lattice", "library")
TARGET_DOTTED = "autolens.point.solver.shape_solver.AbstractSolver._initial_triangles"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
LOGL_RTOL = 1.0e-12
POSITIONS_ATOL = 1.0e-10
GRAD_RTOL = 1.0e-10
GRAD_ATOL = 1.0e-12
VMAP_RTOL = 1.0e-12
RATIOS = (
    ("control", "lattice"),
    ("control", "exact"),
    ("exact", "lattice"),
    ("library", "lattice"),
    ("control", "library"),
)


# ---------------------------------------------------------------------------
# The cell's own step-0 tables (NumPy, never staged into a trace)
# ---------------------------------------------------------------------------


def _lattice_coordinates(y_min, y_max, x_min, x_max, scale) -> np.ndarray:
    """The integer (y, x) lattice ``CoordinateArrayTriangles.for_limits_and_scale`` tiles."""
    y_shift = int(2 * y_min / scale)
    x_shift = int(x_min / (HEIGHT_FACTOR * scale))
    coordinates = []
    for y in range(y_shift, int(2 * y_max / scale) + 1):
        for x in range(x_shift - 1, int(x_max / (HEIGHT_FACTOR * scale)) + 2):
            coordinates.append([y, x])
    return np.array(coordinates)


_TABLE_BUILD_S: dict = {}


@functools.lru_cache(maxsize=16)
def _cell_tables(y_min, y_max, x_min, x_max, scale) -> dict:
    """``{"exact": (V, I), "lattice": (V, I), "coordinates": C}`` for one geometry."""
    t0 = time.perf_counter()
    coordinates = _lattice_coordinates(y_min, y_max, x_min, x_max, scale)
    flip = np.where((coordinates[:, 0] + coordinates[:, 1]) % 2 != 0, -1, 1)[:, None]
    scaling = np.array([0.5 * scale, HEIGHT_FACTOR * scale])
    centres = scaling * coordinates + np.array([0.0, 0.0])
    triangles = np.stack(
        (
            centres + flip * np.array([0.0, 0.5 * scale * HEIGHT_FACTOR]),
            centres + flip * np.array([0.5 * scale, -0.5 * scale * HEIGHT_FACTOR]),
            centres + flip * np.array([-0.5 * scale, -0.5 * scale * HEIGHT_FACTOR]),
        ),
        axis=1,
    )
    flat = triangles.reshape(-1, 2)

    exact_vertices, exact_inverse = np.unique(flat, axis=0, return_inverse=True)

    offsets = np.array([[0, 1], [1, -1], [-1, -1]])
    keys = np.stack(
        (
            coordinates[:, None, 0] + flip * offsets[None, :, 0],
            2 * coordinates[:, None, 1] + flip * offsets[None, :, 1],
        ),
        axis=-1,
    ).reshape(-1, 2)
    _, first, lattice_inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    tables = {
        "coordinates": coordinates,
        "exact": (exact_vertices, exact_inverse.reshape(-1, 3)),
        "lattice": (flat[first].copy(), lattice_inverse.reshape(-1, 3)),
    }
    _TABLE_BUILD_S[(y_min, y_max, x_min, x_max, scale)] = time.perf_counter() - t0
    return tables


class _TableTriangles(_coordinate_array.CoordinateArrayTriangles):
    """The library lattice with a cell-supplied ``(vertices, indices)`` table.

    Only ``_vertices_and_indices`` changes; every derived lattice is built as the base
    class by the library's own methods, so steps >= 1 are untouched.
    """

    def __init__(self, coordinates, side_length, table):
        super().__init__(coordinates=coordinates, side_length=side_length)
        self._cell_table = table

    @property
    def _vertices_and_indices(self):
        vertices, indices = self._cell_table
        return jnp.asarray(vertices), jnp.asarray(indices)


def _geometry(solver) -> tuple:
    return (
        float(solver.y_min),
        float(solver.y_max),
        float(solver.x_min),
        float(solver.x_max),
        float(solver.scale),
    )


def _control_initial_triangles(self, xp):
    """The phase-2 main path: the flat (3N, 2) per-triangle table, no table attached."""
    return _coordinate_array.CoordinateArrayTriangles.for_limits_and_scale(
        y_min=self.y_min, y_max=self.y_max, x_min=self.x_min, x_max=self.x_max, scale=self.scale
    )


def _table_initial_triangles(kind):
    def initial_triangles(self, xp):
        tables = _cell_tables(*_geometry(self))
        return _TableTriangles(
            coordinates=jnp.array(tables["coordinates"]),
            side_length=self.scale,
            table=tables[kind],
        )

    initial_triangles.__name__ = f"_{kind}_initial_triangles"
    return initial_triangles


_exact_initial_triangles = _table_initial_triangles("exact")
_lattice_initial_triangles = _table_initial_triangles("lattice")

_ROUTE_BODIES = {
    "control": _control_initial_triangles,
    "exact": _exact_initial_triangles,
    "lattice": _lattice_initial_triangles,
    "library": None,
}


def _body_provenance(fn) -> dict:
    """Source text + bytecode hash of *fn*, captured at import (phase-2 provenance fix)."""
    code = fn.__code__
    digest = hashlib.sha256(code.co_code + repr(code.co_consts).encode()).hexdigest()[:16]
    source = inspect.getsource(fn)
    return {"name": fn.__name__, "body": source, "bytecode_sha256": digest}


_BODY_PROVENANCE = {
    "control": _body_provenance(_control_initial_triangles),
    "table_factory": _body_provenance(_table_initial_triangles),
    "tables": _body_provenance(_cell_tables.__wrapped__),
    "table_triangles": {
        "body": inspect.getsource(_TableTriangles),
        "bytecode_sha256": hashlib.sha256(
            _TableTriangles._vertices_and_indices.fget.__code__.co_code
        ).hexdigest()[:16],
    },
}


@contextlib.contextmanager
def route_injected(route: str):
    """Swap ``AbstractSolver._initial_triangles`` for *route* and record step-0 rows.

    Yields ``{"calls": n, "step0_rows": [...]}``: how often the route's tiling ran while
    tracing and the row count of the grid each step 0 deflected. ``library`` wraps the
    installed method only to count and record. Both patches are restored by identity.
    """
    cls = _shape_solver.AbstractSolver
    original = cls.__dict__["_initial_triangles"]
    original_plane_grid = cls.__dict__["_plane_grid"]
    params = list(inspect.signature(original).parameters)
    if params != ["self", "xp"]:
        raise TypeError(
            f"{TARGET_DOTTED} now takes {params}; the substituted bodies take (self, xp). "
            "The library changed under the injection -- update this harness, not the library."
        )
    body = _ROUTE_BODIES[route] or original
    record = {"calls": 0, "step0_rows": []}
    armed = {"next": False}

    def counted(self, xp):
        record["calls"] += 1
        armed["next"] = True
        return body(self, xp)

    def plane_grid(self, tracer, grid, xp, plane_redshift=None):
        if armed["next"]:
            record["step0_rows"].append(int(np.shape(grid)[0]))
            armed["next"] = False
        return original_plane_grid(self, tracer, grid, xp, plane_redshift)

    cls._initial_triangles = counted
    cls._plane_grid = plane_grid
    try:
        yield record
    finally:
        cls._initial_triangles = original
        cls._plane_grid = original_plane_grid
        if (
            cls.__dict__["_initial_triangles"] is not original
            or cls.__dict__["_plane_grid"] is not original_plane_grid
        ):
            raise RuntimeError(f"{TARGET_DOTTED} was not restored by identity")


# ---------------------------------------------------------------------------
# Structural readouts
# ---------------------------------------------------------------------------


def _rss_bytes() -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _thread_count() -> int | None:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


def _memory_analysis(compiled) -> dict | str:
    try:
        stats = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001 -- optional on some backends
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
    except Exception as exc:  # noqa: BLE001 -- cost analysis is optional on some backends
        return f"unavailable: {type(exc).__name__}"


_HLO_CONSTANT = re.compile(r"=\s*(\w+)\[([\d,]*)\][^=]*\bconstant\(")


def _hlo_readout(compiled) -> dict:
    hlo = compiled.as_text()
    big = 0
    for m in _HLO_CONSTANT.finditer(hlo):
        dims = [int(d) for d in m.group(2).split(",") if d]
        if dims and int(np.prod(dims)) >= 1000:
            big += 1
    return {
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest()[:16],
        "hlo_lines": hlo.count("\n"),
        "hlo_large_constants": big,
    }


_THREADS_AFTER_FIRST_COMPILE: dict = {}


def _compile_route(route: str, factory, example, *, label: str, expected_rows=None) -> dict:
    """Lower a FRESH closure under *route* after ``jax.clear_caches()``; compile it."""
    jax.clear_caches()
    fn = factory()
    rss0 = _rss_bytes()
    with route_injected(route) as record:
        t0 = time.perf_counter()
        lowered = jax.jit(fn).lower(example)
        lower_s = time.perf_counter() - t0
    if route != "library" and record["calls"] == 0:
        raise RuntimeError(
            f"FAIL [{label}/{route}]: the substituted {TARGET_DOTTED} never ran while tracing."
        )
    if expected_rows is not None and route != "library":
        bad = [r for r in record["step0_rows"] if r != expected_rows[route]]
        if bad or not record["step0_rows"]:
            raise RuntimeError(
                f"FAIL [{label}/{route}]: step-0 rows {record['step0_rows']} but the route's "
                f"table has {expected_rows[route]} rows -- the injection did not take."
            )
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    rss1 = _rss_bytes()
    t0 = time.perf_counter()
    first = block(compiled(example))
    first_call_s = time.perf_counter() - t0
    if not _THREADS_AFTER_FIRST_COMPILE:
        _THREADS_AFTER_FIRST_COMPILE["threads"] = _thread_count()
        _THREADS_AFTER_FIRST_COMPILE["label"] = f"{label}/{route}"
    readout = _hlo_readout(compiled)
    print(
        f"  [{label}/{route}] lower {lower_s:6.1f} s  compile {compile_s:6.1f} s  "
        f"first {first_call_s * 1e3:8.2f} ms  trace_calls {record['calls']:3d}  "
        f"step0_rows {sorted(set(record['step0_rows']))}"
    )
    return {
        "executable": compiled,
        "first": first,
        "record": {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "first_call_s": float(first_call_s),
            "trace_calls": int(record["calls"]),
            "step0_rows": record["step0_rows"],
            "flops": _flops(compiled),
            "memory_analysis": _memory_analysis(compiled),
            "rss_before_lower_bytes": rss0,
            "rss_after_compile_bytes": rss1,
            "rss_delta_bytes": (rss1 - rss0) if rss0 is not None and rss1 is not None else None,
            **readout,
        },
    }


def _constant_folding_probe() -> dict:
    """Does XLA fold constants in THIS process? Read from the optimised HLO, not the flags.

    The probe is a dot of two 1000-element constants. Elementwise arithmetic on
    constants (``c * 2 + 1``, ``sin(c)``) comes back as a literal whether or not the
    constant-folding pass is disabled, so it cannot tell the two modes apart; the
    dot is live with ``--xla_disable_hlo_passes=constant_folding`` and folded to a
    literal without it (measured on the laptop, jax 0.10.2 CPU, 2026-09-24).
    """
    c = np.linspace(0.1, 1.0, 1000)
    compiled = jax.jit(lambda: jnp.asarray(c) @ jnp.asarray(c)).lower().compile()
    hlo = compiled.as_text()
    entry = hlo[hlo.find("ENTRY") :]
    # XLA may rewrite the dot (e.g. multiply + reduce-window), so "live" is any
    # computation left in the entry; folded, the entry is only a constant.
    live = bool(re.search(r"\b(fusion|dot|multiply|add|reduce|reduce-window)\(", entry))
    return {
        "probe": "jit(lambda: c @ c), c = linspace(0.1, 1, 1000) constant",
        "entry_has_computation": live,
        "folding_ran": not live,
        "entry_hlo": entry[:2000],
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


def _step0_containing_factory(setup: dict):
    """FRESH ``vector -> step-0 containing_indices`` closures (simple plain model)."""
    model = setup["models"]["plain"]

    def factory():
        solver = setup["solver"]()

        def containing(vector):
            instance = model.instance_from_vector(vector=vector, xp=jnp)
            galaxies = list(instance.galaxies)
            tracer = al.Tracer(galaxies=galaxies)
            centre = instance.galaxies.source.point_0.centre
            shape = Point(centre[0], centre[1])
            step = next(solver.steps(tracer=tracer, shape=shape, xp=jnp))
            return step.plane_triangles.containing_indices(shape=shape)

        return containing

    return factory


def _expected_rows(setup: dict) -> dict:
    """Step-0 row count of each substituted route for this model's solver geometry."""
    solver = setup["solver"]()
    geometry = _geometry(solver)
    tables = _cell_tables(*geometry)
    n = tables["coordinates"].shape[0]
    library_lattice = _coordinate_array.CoordinateArrayTriangles.for_limits_and_scale(
        y_min=solver.y_min,
        y_max=solver.y_max,
        x_min=solver.x_min,
        x_max=solver.x_max,
        scale=solver.scale,
    )
    if not np.array_equal(np.asarray(library_lattice.coordinates), tables["coordinates"]):
        raise RuntimeError("the cell's lattice coordinates differ from the library's")
    return {
        "geometry": list(geometry),
        "n_triangles": int(n),
        "rows": {
            "control": 3 * int(n),
            "exact": int(tables["exact"][0].shape[0]),
            "lattice": int(tables["lattice"][0].shape[0]),
        },
    }


def _table_provenance(setup: dict) -> dict:
    """Table build time, cache-hit time and size, cell and (if present) library."""
    solver = setup["solver"]()
    geometry = _geometry(solver)
    tables = _cell_tables(*geometry)
    t0 = time.perf_counter()
    _cell_tables(*geometry)
    hit_s = time.perf_counter() - t0
    out = {
        "cell_build_ms": 1e3 * _TABLE_BUILD_S.get(geometry, float("nan")),
        "cell_cache_hit_us": 1e6 * hit_s,
        "exact_table_mb": (tables["exact"][0].nbytes + tables["exact"][1].nbytes) / 1e6,
        "lattice_table_mb": (tables["lattice"][0].nbytes + tables["lattice"][1].nbytes) / 1e6,
    }
    builder = getattr(_coordinate_array, "static_vertex_table", None)
    if builder is None:
        out["library_static_vertex_table"] = "absent (library predates phase 3)"
        return out
    builder.cache_clear()
    t0 = time.perf_counter()
    vertices, indices = builder(*geometry)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    builder(*geometry)
    hit = time.perf_counter() - t0
    lattice_vertices = tables["lattice"][0]
    out["library_static_vertex_table"] = {
        "build_ms": 1e3 * cold,
        "cache_hit_us": 1e6 * hit,
        "rows": int(vertices.shape[0]),
        "mb": (vertices.nbytes + indices.nbytes) / 1e6,
        "equals_cell_lattice_table": bool(
            np.array_equal(vertices, lattice_vertices)
            and np.array_equal(indices, tables["lattice"][1])
        ),
    }
    return out


# ---------------------------------------------------------------------------
# One interleaved timing row
# ---------------------------------------------------------------------------


def _timed_call(executable, argument):
    t0 = time.perf_counter()
    out = block(executable(argument))
    return time.perf_counter() - t0, out


def _run_row(label, factory, stream, *, expected_rows, batch=None) -> dict:
    """Compile four routes, warm, interleave; return timings, values and structure."""
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
        route: _compile_route(
            route, factory, arguments[0], label=label, expected_rows=expected_rows
        )
        for route in ROUTES
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
    ratios = {
        f"{a}_over_{b}": _median_ratio(times[a], times[b], BOOTSTRAP_SEED + i)
        for i, (a, b) in enumerate(RATIOS)
    }
    paired = np.asarray(per_round["control"]) / np.asarray(per_round["lattice"])
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
        "ratios": ratios,
        "paired_round_ratio_control_over_lattice": {
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
        "likelihood_gate": {
            f"control_vs_{route}": _logl_gate(value_arrays["control"], value_arrays[route])
            for route in ROUTES[1:]
        },
    }
    if batch is None:
        row["fiducial_log_likelihood"] = {
            route: float(values[route][0]) for route in ROUTES if 0 in values[route]
        }
    r = ratios["control_over_lattice"]
    print(
        "  median ms  "
        + "  ".join(f"{route} {stats[route]['median_ms']:9.3f}" for route in ROUTES)
        + f"   control/lattice {r['ratio']:.3f} [{r['ci90_low']:.3f}, {r['ci90_high']:.3f}]"
    )
    return row


def _logl_gate(a, b) -> dict:
    delta = _array_delta(a, b)
    delta["pass"] = bool(delta["nan_pattern_equal"] and delta["max_rel_delta"] <= LOGL_RTOL)
    delta["rtol"] = LOGL_RTOL
    return delta


def _positions_gate(label, factory, stream, *, expected_rows) -> dict:
    print(f"\n--- positions gate {label} ---")
    arguments = [jnp.asarray(v) for v in stream[:N_GATE]]
    per_route = {}
    compile_records = {}
    for route in ROUTES:
        compiled = _compile_route(
            route, factory, arguments[0], label=f"{label}_positions", expected_rows=expected_rows
        )
        compile_records[route] = compiled["record"]
        per_route[route] = [
            [np.asarray(p, dtype=float) for p in block(compiled["executable"](a))]
            for a in arguments
        ]

    def finite_sorted(p):
        p = p[np.all(np.isfinite(p), axis=-1)]
        return p[np.lexsort((p[:, 1], p[:, 0]))] if p.size else p.reshape(0, 2)

    counts = {
        route: [[int(np.isfinite(p).all(axis=-1).sum()) for p in inst] for inst in rows]
        for route, rows in per_route.items()
    }
    pairs = {}
    for route in ROUTES[1:]:
        same_counts = counts[route] == counts["control"]
        max_delta = 0.0
        if same_counts:
            for inst_a, inst_b in zip(per_route["control"], per_route[route]):
                for pa, pb in zip(inst_a, inst_b):
                    fa, fb = finite_sorted(pa), finite_sorted(pb)
                    if fa.size:
                        max_delta = max(max_delta, float(np.abs(fa - fb).max()))
        flat_a = np.concatenate([np.ravel(p) for inst in per_route["control"] for p in inst])
        flat_b = np.concatenate([np.ravel(p) for inst in per_route[route] for p in inst])
        pairs[f"control_vs_{route}"] = {
            "image_counts_identical": bool(same_counts),
            "max_abs_delta_finite_sorted": max_delta if same_counts else None,
            "padded_array_equal_nan_aware": bool(np.array_equal(flat_a, flat_b, equal_nan=True)),
            "pass": bool(same_counts and max_delta <= POSITIONS_ATOL),
            "atol": POSITIONS_ATOL,
        }
    return {
        "n_instances": len(arguments),
        "compile": compile_records,
        "positions": {
            route: [[p.tolist() for p in inst] for inst in per_route[route]] for route in ROUTES
        },
        "finite_positions": counts,
        "pairs": pairs,
    }


def _grad_gate(label, factory, stream, *, expected_rows) -> dict:
    print(f"\n--- gradient gate {label} ---")
    arguments = [jnp.asarray(v) for v in stream[:N_GATE]]

    def grad_factory():
        return jax.grad(factory())

    grads = {}
    records = {}
    for route in ROUTES:
        compiled = _compile_route(
            route, grad_factory, arguments[0], label=f"{label}_grad", expected_rows=expected_rows
        )
        records[route] = compiled["record"]
        grads[route] = np.stack(
            [np.asarray(block(compiled["executable"](a)), dtype=float) for a in arguments]
        )
    pairs = {}
    for route in ROUTES[1:]:
        delta = _array_delta(grads["control"], grads[route])
        delta["allclose"] = bool(
            np.allclose(grads["control"], grads[route], rtol=GRAD_RTOL, atol=GRAD_ATOL)
        )
        pairs[f"control_vs_{route}"] = delta
    return {
        "n_instances": len(arguments),
        "rtol": GRAD_RTOL,
        "atol": GRAD_ATOL,
        "compile": records,
        "all_finite": {route: bool(np.isfinite(g).all()) for route, g in grads.items()},
        "any_nonzero_per_instance": {
            route: [bool(np.any(row != 0.0)) for row in g] for route, g in grads.items()
        },
        "gradients": {route: g.tolist() for route, g in grads.items()},
        "pairs": pairs,
    }


def _step0_gate(factory, stream, *, expected_rows) -> dict:
    print("\n--- step-0 containing_indices gate simple_plain ---")
    arguments = [jnp.asarray(v) for v in stream[:N_GATE]]
    sets = {}
    for route in ROUTES:
        compiled = _compile_route(
            route, factory, arguments[0], label="simple_plain_step0", expected_rows=expected_rows
        )
        sets[route] = []
        for a in arguments:
            idx = np.asarray(block(compiled["executable"](a)))
            sets[route].append(sorted(int(i) for i in idx[idx >= 0]))
    return {
        "n_instances": len(arguments),
        "sets": sets,
        "pairs": {
            f"control_vs_{route}": {"identical": sets[route] == sets["control"]}
            for route in ROUTES[1:]
        },
    }


def _vmap_gate(vmap_row: dict, scalar_row: dict) -> dict:
    """``vmap`` batch values vs the scalar row's per-instance values, per route."""
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
        delta["allclose"] = bool(
            np.allclose(batched, reference, rtol=VMAP_RTOL, atol=0.0, equal_nan=True)
        )
        delta["n_compared"] = len(batched)
        within[route] = delta
    return {"vmap_vs_scalar_within_route": within, "across_routes": vmap_row["likelihood_gate"]}


def _library_label(rows: dict, expected: dict) -> dict:
    per_row = {}
    for label, row in rows.items():
        model_name = label.split("_")[0]
        rows_expected = expected[model_name]["rows"]
        library_rows = sorted(set(row["compile"]["library"]["step0_rows"]))
        match = "neither"
        if len(library_rows) == 1:
            for route in ("control", "exact", "lattice"):
                if library_rows[0] == rows_expected[route]:
                    match = route
        hashes = {route: row["compile"][route]["hlo_sha256"] for route in ROUTES}
        per_row[label] = {
            "library_step0_rows": library_rows,
            "expected_rows": rows_expected,
            "library_matches": match,
            "library_hlo_identical_to": [
                route for route in ROUTES[:3] if hashes[route] == hashes["library"]
            ],
        }
    labels = {entry["library_matches"] for entry in per_row.values()}
    overall = labels.pop() if len(labels) == 1 else "inconsistent"
    return {"per_row": per_row, "library_matches": overall}


# ===========================================================================
# Run
# ===========================================================================

constant_folding_probe = _constant_folding_probe()
print(
    f"constant folding: requested={'ENABLED' if CONSTANT_FOLDING else 'disabled'}  "
    f"probe folding_ran={constant_folding_probe['folding_ran']}  XLA_FLAGS={os.environ.get('XLA_FLAGS')}"
)
constant_folding_probe["agrees_with_request"] = (
    CONSTANT_FOLDING == constant_folding_probe["folding_ran"]
)
if not constant_folding_probe["agrees_with_request"]:
    message = (
        f"constant folding requested={CONSTANT_FOLDING} but the HLO probe says "
        f"folding_ran={constant_folding_probe['folding_ran']} (XLA_FLAGS={os.environ.get('XLA_FLAGS')})"
    )
    # The probe was calibrated on the CPU backend; elsewhere record, do not refuse.
    if jax.default_backend() == "cpu":
        raise RuntimeError(f"FAIL: {message}")
    print(f"WARNING: {message} -- probe calibrated on CPU only; recorded in the JSON")

rows: dict[str, dict] = {}
gates: dict[str, dict] = {}
model_configurations: dict[str, dict] = {}
free_parameters: dict[str, int] = {}
expected: dict[str, dict] = {}
tables_provenance: dict[str, dict] = {}

for model_name in MODELS:
    setup = _simple_setup() if model_name == "simple" else _cluster_setup()
    model_configurations[model_name] = setup["configuration"]
    expected[model_name] = _expected_rows(setup)
    tables_provenance[model_name] = _table_provenance(setup)
    expected_rows = expected[model_name]["rows"]
    print(
        f"\n=== {model_name}: step-0 rows {expected_rows}; tables {tables_provenance[model_name]}"
    )
    # Load-bearing for the gradient gate: without it jax.grad of this likelihood
    # comes back identically ZERO (forward values unaffected).
    for _model in setup["models"].values():
        register_model_pytrees(_model)
    for variant in ("solved", "plain"):
        model = setup["models"][variant]
        free_parameters[f"{model_name}_{variant}"] = int(model.prior_count)
        stream = _vector_stream(model, N_STREAM, SEED + (0 if variant == "solved" else 1))
        factory = _likelihood_factory(setup, variant)
        label = f"{model_name}_{variant}"
        rows[label] = _run_row(label, factory, stream, expected_rows=expected_rows)
        if variant == "solved":
            gates[f"{label}_positions"] = _positions_gate(
                label, _positions_factory(setup, variant), stream, expected_rows=expected_rows
            )
        if label == "simple_solved":
            gates[f"{label}_grad"] = _grad_gate(label, factory, stream, expected_rows=expected_rows)
            vlabel = f"{label}_vmap{VMAP_BATCH}"
            rows[vlabel] = _run_row(
                vlabel,
                lambda f=factory: jax.vmap(f()),
                stream,
                expected_rows=expected_rows,
                batch=VMAP_BATCH,
            )
            gates[vlabel] = _vmap_gate(rows[vlabel], rows[label])
        if label == "simple_plain":
            gates["simple_plain_step0_containing"] = _step0_gate(
                _step0_containing_factory(setup), stream, expected_rows=expected_rows
            )

library_label = _library_label(rows, expected)


def _gate_summary() -> dict:
    """One flat verdict table: every gate, control vs each other route."""
    summary = {}
    for label, row in rows.items():
        for pair, g in row["likelihood_gate"].items():
            summary[f"{label}.log_likelihood.{pair}"] = {
                "pass": g["pass"],
                "bit_identical": g["bit_identical"],
                "max_rel_delta": g["max_rel_delta"],
            }
    for label, gate in gates.items():
        if label.endswith("_positions"):
            for pair, g in gate["pairs"].items():
                summary[f"{label}.{pair}"] = {
                    "pass": g["pass"],
                    "image_counts_identical": g["image_counts_identical"],
                    "max_abs_delta": g["max_abs_delta_finite_sorted"],
                }
        elif label.endswith("_grad"):
            nonzero = all(all(v) for v in gate["any_nonzero_per_instance"].values())
            for pair, g in gate["pairs"].items():
                summary[f"{label}.{pair}"] = {
                    "pass": g["allclose"] and all(gate["all_finite"].values()) and nonzero,
                    "non_zero": nonzero,
                    "bit_identical": g["bit_identical"],
                    "max_rel_delta": g["max_rel_delta"],
                }
        elif label.endswith("_step0_containing"):
            for pair, g in gate["pairs"].items():
                summary[f"{label}.{pair}"] = {"pass": g["identical"]}
        else:
            for route, g in gate["vmap_vs_scalar_within_route"].items():
                summary[f"{label}.vmap_vs_scalar.{route}"] = {
                    "pass": g["allclose"],
                    "bit_identical": g["bit_identical"],
                    "max_abs_delta": g["max_abs_delta"],
                }
    return summary


gate_summary = _gate_summary()
all_gates_pass = all(v["pass"] for v in gate_summary.values())

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
    "cell": "static_lattice_ab",
    "issue": "PyAutoLabs/PyAutoArray#568",
    "phase": 3,
    "config_name": config_name,
    "quotable": not config_name.startswith(("laptop", "local")),
    "host_note": (
        "laptop witness: WSL2, interactive load, NOT quotable -- ratios indicative only"
        if config_name.startswith(("laptop", "local"))
        else None
    ),
    "quick": QUICK,
    "models": list(MODELS),
    "constant_folding": {
        "requested": CONSTANT_FOLDING,
        "xla_flags_requested": _XLA_FLAGS_REQUESTED,
        "xla_flags_effective": os.environ.get("XLA_FLAGS"),
        "probe": constant_folding_probe,
    },
    "library_versions": {
        "autolens": al.__version__,
        "autoarray": aa.__version__,
        "autofit": af.__version__,
    },
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    "source_revisions": source_revisions(_ROOT),
    "autoarray_imported_from": str(Path(aa.__file__).resolve().parent.parent),
    "autolens_imported_from": str(Path(al.__file__).resolve().parent.parent),
    "target": TARGET_DOTTED,
    "route_bodies": _BODY_PROVENANCE,
    "body_provenance_captured": "import time",
    "expected_step0_rows": expected,
    "tables": tables_provenance,
    "device": device,
    "jax_devices": [str(d) for d in jax.devices()],
    "machine": machine_info_dict(),
    "thread_environment": thread_environment(),
    "xla_threads_observed_after_first_compile": _THREADS_AFTER_FIRST_COMPILE,
    "threads_at_end": _thread_count(),
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
        "gates": {
            "log_likelihood_rtol": LOGL_RTOL,
            "positions_atol": POSITIONS_ATOL,
            "grad_rtol": GRAD_RTOL,
            "grad_atol": GRAD_ATOL,
            "vmap_rtol": VMAP_RTOL,
            "cluster": "log-likelihood and positions only (no cluster grad/vmap/step-0 sets)",
        },
        "pytrees": "autofit.jax.register_model(model) for every model before any trace",
    },
    "free_parameters": free_parameters,
    "model_configuration": model_configurations,
    "library_label": library_label,
    "library_matches": library_label["library_matches"],
    "all_gates_pass": all_gates_pass,
    "gate_summary": gate_summary,
    "rows": rows,
    "gates": gates,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_image",
    default_basename="static_lattice_ab_local",
    cell="static_lattice_ab" + ("_constant_folding" if CONSTANT_FOLDING else ""),
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2))

# ---------------------------------------------------------------------------
# PNG: per-row boxplots, the four routes
# ---------------------------------------------------------------------------
colours = {"control": "#C44E52", "exact": "#DD8452", "lattice": "#4C72B0", "library": "#8C8C8C"}
fig, axes = plt.subplots(
    nrows=1, ncols=len(rows), figsize=(4.4 * len(rows), 5.0), constrained_layout=True
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
    r = row["ratios"]["control_over_lattice"]
    e = row["ratios"]["control_over_exact"]
    ax.set_title(
        f"{label}\ncontrol/lattice {r['ratio']:.2f}x [{r['ci90_low']:.2f}, {r['ci90_high']:.2f}]"
        f"\ncontrol/exact {e['ratio']:.2f}x",
        fontsize=9,
    )
    ax.set_ylim(bottom=0)
fig.suptitle(
    f"PointSolver static step-0 lattice A/B ({config_name}"
    f"{', constant folding ON' if CONSTANT_FOLDING else ''}); "
    f"library_matches={library_label['library_matches']}; all_gates_pass={all_gates_pass}; "
    "whiskers p10/p90",
    fontsize=10,
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 110)
print(
    f"STATIC-LATTICE A/B  ({config_name}, rounds={N_ROUNDS}, calls/round={N_CALLS}, "
    f"constant_folding={'ON' if CONSTANT_FOLDING else 'off'})"
)
print("=" * 110)
print(
    f"  {'row':<22} {'route':<8} {'median':>9} {'p10':>9} {'p90':>9} {'min':>9}"
    f" {'step0':>6} {'MFLOP':>8} {'temp_MB':>8} {'compile_s':>9}"
)
for label, row in rows.items():
    for route in ROUTES:
        s = row["stats"][route]
        c = row["compile"][route]
        flops = c["flops"] / 1e6 if isinstance(c["flops"], float) else float("nan")
        mem = c["memory_analysis"]
        temp = (
            mem.get("temp_size_in_bytes", float("nan")) / 1e6
            if isinstance(mem, dict)
            else float("nan")
        )
        step0 = c["step0_rows"][0] if c["step0_rows"] else -1
        print(
            f"  {label:<22} {route:<8} {s['median_ms']:9.3f} {s['p10_ms']:9.3f} {s['p90_ms']:9.3f}"
            f" {s['min_ms']:9.3f} {step0:6d} {flops:8.2f} {temp:8.2f} {c['compile_s']:9.1f}"
        )
    for name, r in row["ratios"].items():
        print(
            f"  {'':<22} {name:<22} {r['ratio']:.3f} [90% {r['ci90_low']:.3f}, {r['ci90_high']:.3f}]"
        )
    print(f"  {'':<22} MDI {row['minimum_detectable_improvement']:.3f}")
print("-" * 110)
print(f"  library_matches: {library_label['library_matches']}")
print(f"  all_gates_pass: {all_gates_pass}")
for name, verdict in gate_summary.items():
    if not verdict["pass"]:
        print(f"  GATE NOT PASSED: {name}: {verdict}")
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
