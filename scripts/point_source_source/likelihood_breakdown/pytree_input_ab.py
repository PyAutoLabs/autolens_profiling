"""
JAX Profiling: Source-Plane Point-Source Pytree-Input A/B (phase 2a)
====================================================================

An interleaved, in-process A/B of *how the parameters enter* the production
source-plane point-source likelihood (autolens_profiling #322, single-source
only). Phase 1 (``source_plane.py``) put the fused call at the JAX dispatch
floor and found ~0.07-0.09 ms of it to be the fixed cost of passing the
``ModelInstance`` pytree: on every call JAX flattens the registered
``ModelInstance`` in Python (``autofit/jax/pytrees.py``) before the executable
runs. This cell measures that cost against two flat entry points, so the
phase-2b lever is ranked on clean numbers.

Routes (per lane)
-----------------

- ``pytree`` — production: ``analysis.log_likelihood_function(instance=params)``
  with the registered ``ModelInstance`` pytree as the traced argument (exactly
  the phase-1 fused control).
- ``flat_vector`` — the physical parameter vector as the traced argument and
  ``model.instance_from_vector(vector, xp=jnp)`` inside the trace (the
  image-plane cells' route, as ``autofit``'s JAX ``Fitness`` does).
- ``flat_leaves`` — the pytree's leaves as a tuple argument, rebuilt with
  ``jax.tree_util.tree_unflatten(treedef, leaves)`` inside the trace. It isolates
  the Python flatten of the ``ModelInstance`` (paid by ``pytree`` only) from the
  ``instance_from_vector`` work (paid by ``flat_vector`` only, at trace time).

Lanes: ``solved`` = ``PointSolved`` + ``FitPositionsSourceSolved`` (the adopted
default); ``plain`` = ``PointFlux`` + ``FitPositionsSource``.

Rows (per lane, each timing the three routes interleaved): ``forward`` (the
likelihood), ``value_and_grad`` (``jax.value_and_grad`` w.r.t. the route's own
argument) and ``floor`` (a jitted sum over the argument's leaves — phase 1's
dispatch probe — so argument handling is separated from compute).

The A/B protocol
----------------

Copied from ``scripts/point_source_image/likelihood_breakdown/vertex_dedup_ab.py``:
every route lowers a FRESH closure built by a factory (fresh ``AnalysisPoint``)
after ``jax.clear_caches()``, so no route can be served another's trace; lower,
compile and first-call times are recorded. The three executables are then warmed
(>= 3 calls) and timed in ``--rounds`` round-robin rounds with the route order
rotated each round and ``--calls`` calls per route per round, walking a
fixed-seed 16-instance parameter stream (entry 0 the prior medians, the rest
``vector_from_unit_vector(U(0.25, 0.75))``) that feeds every route in the
route's own argument form. Every call is ``block_until_ready``'d; per-instance
determinism within a route is recorded. Per route: median / p10 / p90 / min /
max ms; the median ratios ``pytree/flat_vector``, ``pytree/flat_leaves`` and
``flat_leaves/flat_vector`` with bootstrap 90 % intervals; the absolute median
ms saved; the minimum detectable improvement is the ``pytree`` row's
``(p90 - p10) / median``. The timed object is the ``Compiled`` executable
(``jax.jit(fn).lower(x).compile()``), which takes the same C++ dispatch path
as a ``jax.jit`` call and flattens custom pytree nodes in Python on each call.

Asserts (hard failures): all routes agree on log L per streamed instance at
rtol 1e-10 (forward and ``value_and_grad`` values); every route's gradient is
finite AND non-zero (an unregistered model makes ``jax.grad`` silently zero).

Phase-2b rule (issue #322; applies to the RAL CPU row): **go** if
``pytree - flat_vector`` >= 0.05 ms AND >= 15 % of the fused (``pytree``)
forward call; else no-go. The JSON evaluates it per lane for every config;
only ``hpc_ral_cpu_fp64`` decides.

Output
------

``results/breakdown/point_source_source/pytree_input_ab_<config_name>.{json,png}``.
The JSON deliberately carries no top-level ``autolens_version`` (versions live
under ``library_versions``): this is a note-backed A/B, not a README auto-table
breakdown row.
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
import json

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
_cell_parser.add_argument("--n-stream", type=int, default=16)
_cell_parser.add_argument("--warm", type=int, default=3)
_cell_parser.add_argument("--seed", type=int, default=322)
_cell_parser.add_argument("--quick", action="store_true", help="rounds=5, calls=5")
_args = _cli.parse_cell_args(_cell_parser)

QUICK = bool(_args.quick)
N_ROUNDS = _args.rounds if _args.rounds is not None else (5 if QUICK else 20)
N_CALLS = _args.calls if _args.calls is not None else (5 if QUICK else 20)
N_STREAM = _args.n_stream
N_WARM = max(3, _args.warm)
SEED = _args.seed
if N_ROUNDS < 1 or N_CALLS < 1 or N_STREAM < 1:
    raise SystemExit("--rounds, --calls and --n-stream must be positive")

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402

INSTRUMENT = "simple"
ROUTES = ("pytree", "flat_vector", "flat_leaves")
LANES = ("solved", "plain")
ROW_KINDS = ("forward", "value_and_grad", "floor")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
AGREEMENT_RTOL = 1.0e-10
#: Phase-2b go/no-go thresholds (issue #322).
GO_MIN_SAVED_MS = 0.05
GO_MIN_FRACTION = 0.15

_THREADS_AFTER_FIRST_COMPILE: dict = {}


def _thread_count() -> int | None:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Model / analysis builders — copied from source_plane.py (a flat script)
# ---------------------------------------------------------------------------


def _mass_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    return mass


def _model(*, solved: bool):
    lens = af.Model(al.Galaxy, redshift=0.5, mass=_mass_model())
    if solved:
        point = af.Model(al.ps.PointSolved)
    else:
        point = af.Model(al.ps.PointFlux)
        point.centre.centre_0 = af.GaussianPrior(mean=0.07, sigma=0.005)
        point.centre.centre_1 = af.GaussianPrior(mean=0.07, sigma=0.005)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


dataset_path = _ROOT / "dataset" / "point_source" / INSTRUMENT
auto_simulate_if_missing(
    dataset_path,
    dataset_type="point_source",
    instrument=INSTRUMENT,
    workspace_root=_ROOT,
)
dataset = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")


def _analysis(fit_positions_cls, *, use_jax: bool):
    return al.AnalysisPoint(
        dataset=dataset,
        solver=None,
        fit_positions_cls=fit_positions_cls,
        use_jax=use_jax,
    )


MODELS = {"solved": _model(solved=True), "plain": _model(solved=False)}
FIT_CLS = {"solved": al.FitPositionsSourceSolved, "plain": al.FitPositionsSource}
# Load-bearing: without registration the ModelInstance is not a pytree and
# jax.grad w.r.t. it is silently all-zero.
for _model_obj in MODELS.values():
    register_model_pytrees(_model_obj)


# ---------------------------------------------------------------------------
# Statistics (verbatim protocol from vertex_dedup_ab.py)
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


def _flops(compiled):
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list | tuple):
            cost = cost[0] if cost else {}
        value = (cost or {}).get("flops")
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001 — cost analysis is optional on some backends
        return f"unavailable: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# Instance streams, one per lane, in every route's argument form
# ---------------------------------------------------------------------------


def _vector_stream(model, n: int, seed: int) -> np.ndarray:
    """Fixed-seed physical vectors: entry 0 the prior medians, the rest U(0.25,0.75) draws."""
    rng = np.random.default_rng(seed)
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for _ in range(n - 1):
        unit = rng.uniform(0.25, 0.75, size=model.prior_count)
        vectors.append(np.asarray(model.vector_from_unit_vector(unit), dtype=float))
    return np.stack(vectors)


def _route_arguments(model, stream: np.ndarray) -> tuple[dict, object]:
    """The same parameter stream as pytree / flat vector / leaves-tuple arguments."""
    pytrees = [
        jax.tree_util.tree_map(jnp.asarray, model.instance_from_vector(vector=list(v)))
        for v in stream
    ]
    treedef = jax.tree_util.tree_structure(pytrees[0])
    for tree in pytrees[1:]:
        if jax.tree_util.tree_structure(tree) != treedef:
            raise RuntimeError("the ModelInstance treedef changes across the stream")
    return {
        "pytree": pytrees,
        "flat_vector": [jnp.asarray(v) for v in stream],
        "flat_leaves": [tuple(jax.tree_util.tree_leaves(tree)) for tree in pytrees],
    }, treedef


# ---------------------------------------------------------------------------
# Route factories: FRESH closures (and a fresh AnalysisPoint) per call
# ---------------------------------------------------------------------------


def _likelihood_factory(lane: str, route: str, treedef):
    model = MODELS[lane]

    def factory():
        analysis = _analysis(FIT_CLS[lane], use_jax=True)
        if route == "pytree":

            def log_likelihood(params):
                return analysis.log_likelihood_function(instance=params)

        elif route == "flat_vector":

            def log_likelihood(vector):
                instance = model.instance_from_vector(vector=vector, xp=jnp)
                return analysis.log_likelihood_function(instance=instance)

        else:

            def log_likelihood(leaves):
                instance = jax.tree_util.tree_unflatten(treedef, list(leaves))
                return analysis.log_likelihood_function(instance=instance)

        return log_likelihood

    return factory


def _floor_factory(route: str):
    def factory():
        if route == "flat_vector":

            def floor(vector):
                return jnp.sum(vector)

        else:

            def floor(argument):
                return sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(argument))

        return floor

    return factory


def _row_factory(kind: str, lane: str, route: str, treedef):
    if kind == "floor":
        return _floor_factory(route)
    base = _likelihood_factory(lane, route, treedef)
    if kind == "forward":
        return base
    return lambda: jax.value_and_grad(base())


def _value_of(kind: str, out) -> np.ndarray:
    return np.asarray(out[0] if kind == "value_and_grad" else out, dtype=float)


# ---------------------------------------------------------------------------
# Compile + one interleaved timing row
# ---------------------------------------------------------------------------


def _compile_route(route: str, factory, example, *, label: str) -> dict:
    """Lower a FRESH closure after ``jax.clear_caches()``; compile it; first call."""
    jax.clear_caches()
    fn = factory()
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(example)
    lower_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    first = block(compiled(example))
    first_call_s = time.perf_counter() - t0
    if not _THREADS_AFTER_FIRST_COMPILE:
        _THREADS_AFTER_FIRST_COMPILE["threads"] = _thread_count()
        _THREADS_AFTER_FIRST_COMPILE["label"] = f"{label}/{route}"
    print(
        f"  [{label}/{route}] lower {lower_s * 1e3:8.1f} ms  compile {compile_s * 1e3:8.1f} ms  "
        f"first {first_call_s * 1e3:8.3f} ms"
    )
    return {
        "executable": compiled,
        "first": first,
        "record": {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "first_call_s": float(first_call_s),
            "flops": _flops(compiled),
            "n_argument_leaves": len(jax.tree_util.tree_leaves(example)),
        },
    }


def _timed_call(executable, argument):
    t0 = time.perf_counter()
    out = block(executable(argument))
    return time.perf_counter() - t0, out


def _gradient_check(kind: str, out) -> dict:
    grads = [np.asarray(leaf, dtype=float) for leaf in jax.tree_util.tree_leaves(out[1])]
    finite = all(np.isfinite(g).all() for g in grads)
    l2 = float(np.sqrt(sum(float(np.vdot(g, g)) for g in grads)))
    return {"all_finite": bool(finite), "l2": l2, "n_leaves": len(grads)}


def _run_row(lane: str, kind: str, arguments: dict, treedef) -> dict:
    label = f"{lane}_{kind}"
    print(f"\n--- row {label} ---")
    compiled = {
        route: _compile_route(
            route, _row_factory(kind, lane, route, treedef), arguments[route][0], label=label
        )
        for route in ROUTES
    }
    executables = {route: compiled[route]["executable"] for route in ROUTES}
    n_args = len(arguments["pytree"])

    warm_s = {route: [] for route in ROUTES}
    for route in ROUTES:
        for w in range(N_WARM):
            dt, _ = _timed_call(executables[route], arguments[route][w % n_args])
            warm_s[route].append(dt)

    times = {route: [] for route in ROUTES}
    per_round = {route: [] for route in ROUTES}
    values = {route: {} for route in ROUTES}
    deterministic = {route: True for route in ROUTES}
    gradient_checks: dict[str, dict] = {}
    call_index = 0
    for r in range(N_ROUNDS):
        order = ROUTES[r % len(ROUTES) :] + ROUTES[: r % len(ROUTES)]
        start = call_index
        for route in order:
            round_times = []
            for c in range(N_CALLS):
                k = (start + c) % n_args
                dt, out = _timed_call(executables[route], arguments[route][k])
                round_times.append(dt)
                value = _value_of(kind, out)
                if k in values[route]:
                    if not np.array_equal(values[route][k], value, equal_nan=True):
                        deterministic[route] = False
                else:
                    values[route][k] = value
                    if kind == "value_and_grad":
                        gradient_checks[f"{route}/{k}"] = _gradient_check(kind, out)
            times[route].extend(round_times)
            per_round[route].append(float(np.median(round_times)))
        call_index = start + N_CALLS

    seen = sorted(set.intersection(*(set(values[r]) for r in ROUTES)))
    value_arrays = {route: np.stack([values[route][k] for k in seen]) for route in ROUTES}
    stats = {route: _stats_ms(times[route]) for route in ROUTES}

    # Hard asserts. Floors sum different leaf sets per route, so they carry no
    # cross-route agreement check; likelihood rows must agree per instance.
    agreement = {}
    if kind != "floor":
        for other in ("flat_vector", "flat_leaves"):
            ref, val = value_arrays["pytree"], value_arrays[other]
            max_rel = float(np.max(np.abs(val - ref) / np.maximum(np.abs(ref), 1e-300)))
            agreement[f"pytree_vs_{other}"] = {"max_rel_delta": max_rel, "n": len(seen)}
            np.testing.assert_allclose(
                val,
                ref,
                rtol=AGREEMENT_RTOL,
                err_msg=f"[{label}] {other} disagrees with pytree on log L",
            )
    if kind == "value_and_grad":
        for key, check in gradient_checks.items():
            if not check["all_finite"]:
                raise AssertionError(f"[{label}] gradient {key} contains non-finite values")
            if not check["l2"] > 0.0:
                raise AssertionError(
                    f"[{label}] gradient {key} is identically zero (model unregistered?)"
                )

    pytree_stats = stats["pytree"]
    paired = np.asarray(per_round["pytree"]) / np.asarray(per_round["flat_vector"])
    row = {
        "label": label,
        "lane": lane,
        "kind": kind,
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "route_order": "round-robin, start rotated each round",
        "per_call_ms": {route: [t * 1e3 for t in times[route]] for route in ROUTES},
        "warm_ms": {route: [t * 1e3 for t in warm_s[route]] for route in ROUTES},
        "stats": stats,
        "ratio_pytree_over_flat_vector": _median_ratio(
            times["pytree"], times["flat_vector"], BOOTSTRAP_SEED
        ),
        "ratio_pytree_over_flat_leaves": _median_ratio(
            times["pytree"], times["flat_leaves"], BOOTSTRAP_SEED + 1
        ),
        "ratio_flat_leaves_over_flat_vector": _median_ratio(
            times["flat_leaves"], times["flat_vector"], BOOTSTRAP_SEED + 2
        ),
        "saved_ms_pytree_minus_flat_vector": float(
            pytree_stats["median_ms"] - stats["flat_vector"]["median_ms"]
        ),
        "saved_ms_pytree_minus_flat_leaves": float(
            pytree_stats["median_ms"] - stats["flat_leaves"]["median_ms"]
        ),
        "paired_round_ratio_pytree_over_flat_vector": {
            "median": float(np.median(paired)),
            "p10": float(np.percentile(paired, 10)),
            "p90": float(np.percentile(paired, 90)),
        },
        "minimum_detectable_improvement": float(
            (pytree_stats["p90_ms"] - pytree_stats["p10_ms"]) / pytree_stats["median_ms"]
        ),
        "compile": {route: compiled[route]["record"] for route in ROUTES},
        "instances_compared": seen,
        "values": {route: value_arrays[route].ravel().tolist() for route in ROUTES},
        "deterministic_within_route": deterministic,
        "route_agreement": agreement,
    }
    if gradient_checks:
        row["gradient_checks"] = gradient_checks
        row["gradient_l2_fiducial"] = {
            route: gradient_checks[f"{route}/0"]["l2"]
            for route in ROUTES
            if f"{route}/0" in gradient_checks
        }
    ratio = row["ratio_pytree_over_flat_vector"]
    print(
        "  median ms  "
        + "  ".join(f"{route} {stats[route]['median_ms']:8.4f}" for route in ROUTES)
        + f"   pytree/flat_vector {ratio['ratio']:.3f} "
        f"[{ratio['ci90_low']:.3f}, {ratio['ci90_high']:.3f}]"
    )
    return row


# ===========================================================================
# Run
# ===========================================================================

rows: dict[str, dict] = {}
free_parameters: dict[str, int] = {}
argument_leaves: dict[str, dict] = {}
for lane_index, lane in enumerate(LANES):
    model = MODELS[lane]
    free_parameters[lane] = int(model.prior_count)
    stream = _vector_stream(model, N_STREAM, SEED + lane_index)
    arguments, treedef = _route_arguments(model, stream)
    argument_leaves[lane] = {
        "pytree_leaves": len(arguments["flat_leaves"][0]),
        "vector_length": int(stream.shape[1]),
        "treedef": str(treedef),
    }
    for kind in ROW_KINDS:
        rows[f"{lane}_{kind}"] = _run_row(lane, kind, arguments, treedef)


def _phase2b_rule() -> dict:
    per_lane = {}
    for lane in LANES:
        row = rows[f"{lane}_forward"]
        saved = row["saved_ms_pytree_minus_flat_vector"]
        fused = row["stats"]["pytree"]["median_ms"]
        fraction = saved / fused
        per_lane[lane] = {
            "saved_ms": saved,
            "fused_pytree_ms": fused,
            "fraction_of_fused": fraction,
            "go": bool(saved >= GO_MIN_SAVED_MS and fraction >= GO_MIN_FRACTION),
        }
    return {
        "rule": f"go if pytree - flat_vector >= {GO_MIN_SAVED_MS} ms AND >= "
        f"{GO_MIN_FRACTION:.0%} of the fused forward call",
        "decides_on": "hpc_ral_cpu_fp64 only",
        "per_lane": per_lane,
    }


phase2b_rule = _phase2b_rule()
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
    "cell": "pytree_input_ab",
    "issue": "PyAutoLabs/autolens_profiling#322",
    "config_name": config_name,
    "quick": QUICK,
    "instrument": INSTRUMENT,
    "library_versions": {
        "autolens": al.__version__,
        "autoarray": aa.__version__,
        "autofit": af.__version__,
    },
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    "source_revisions": source_revisions(_ROOT),
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
    "xla_threads_observed_after_first_compile": _THREADS_AFTER_FIRST_COMPILE,
    "threads_at_end": _thread_count(),
    "protocol": {
        "routes": list(ROUTES),
        "lanes": {
            "solved": "PointSolved + FitPositionsSourceSolved",
            "plain": "PointFlux + FitPositionsSource",
        },
        "row_kinds": list(ROW_KINDS),
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "n_stream": N_STREAM,
        "instance_seed": SEED,
        "instance_stream": "entry 0 = prior medians; rest = vector_from_unit_vector(U(0.25,0.75))",
        "arguments": {
            "pytree": "registered ModelInstance (tree_map(jnp.asarray, instance_from_vector))",
            "flat_vector": "physical vector; instance_from_vector(xp=jnp) inside the trace",
            "flat_leaves": "tuple(tree_leaves(pytree)); tree_unflatten(treedef) inside the trace",
        },
        "timed_object": "jax.jit(fn).lower(example).compile() executable",
        "cache_trap_guard": "fresh closure + fresh AnalysisPoint per route; jax.clear_caches()",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": "90%"},
        "asserts": f"log L agreement across routes rtol {AGREEMENT_RTOL}; grads finite and "
        "non-zero for every route and streamed instance",
        "solver": "solver=None; FitPositionsSource never invokes a PointSolver",
        "pytrees": "autofit.jax.register_model(model) for both lanes before any trace",
    },
    "free_parameters": free_parameters,
    "argument_leaves": argument_leaves,
    "phase2b_rule": phase2b_rule,
    "rows": rows,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_source",
    default_basename="pytree_input_ab_local",
    cell="pytree_input_ab",
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2))

# ---------------------------------------------------------------------------
# PNG: per-row boxplots, pytree vs flat_vector vs flat_leaves
# ---------------------------------------------------------------------------
colours = {"pytree": "#C44E52", "flat_vector": "#4C72B0", "flat_leaves": "#55A868"}
fig, axes = plt.subplots(
    nrows=len(LANES),
    ncols=len(ROW_KINDS),
    figsize=(4.4 * len(ROW_KINDS), 4.6 * len(LANES)),
    constrained_layout=True,
)
for i, lane in enumerate(LANES):
    for j, kind in enumerate(ROW_KINDS):
        ax = axes[i, j]
        row = rows[f"{lane}_{kind}"]
        data = [row["per_call_ms"][route] for route in ROUTES]
        box = ax.boxplot(data, whis=(10, 90), showfliers=False, patch_artist=True)
        for patch, route in zip(box["boxes"], ROUTES):
            patch.set_facecolor(colours[route])
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(ROUTES) + 1))
        ax.set_xticklabels(ROUTES, fontsize=8)
        ax.set_ylabel("ms per call")
        rv = row["ratio_pytree_over_flat_vector"]
        rl = row["ratio_pytree_over_flat_leaves"]
        ax.set_title(
            f"{lane} {kind}\npytree/flat_vector {rv['ratio']:.2f}x "
            f"[{rv['ci90_low']:.2f}, {rv['ci90_high']:.2f}]\n"
            f"pytree/flat_leaves {rl['ratio']:.2f}x [{rl['ci90_low']:.2f}, {rl['ci90_high']:.2f}]",
            fontsize=8,
        )
        ax.set_ylim(bottom=0)
fig.suptitle(
    f"Source-plane point-source pytree-input A/B ({config_name}); whiskers p10/p90, "
    f"rounds={N_ROUNDS}, calls/round={N_CALLS}",
    fontsize=10,
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 100)
print(f"PYTREE-INPUT A/B  ({config_name}, rounds={N_ROUNDS}, calls/round={N_CALLS})")
print("=" * 100)
print(f"  {'row':<22} {'route':<12} {'median':>9} {'p10':>9} {'p90':>9} {'min':>9} {'max':>9}")
for label, row in rows.items():
    for route in ROUTES:
        s = row["stats"][route]
        print(
            f"  {label:<22} {route:<12} {s['median_ms']:9.4f} {s['p10_ms']:9.4f} "
            f"{s['p90_ms']:9.4f} {s['min_ms']:9.4f} {s['max_ms']:9.4f}"
        )
    rv = row["ratio_pytree_over_flat_vector"]
    rl = row["ratio_pytree_over_flat_leaves"]
    print(
        f"  {'':<22} pytree/flat_vector {rv['ratio']:.3f} [90% {rv['ci90_low']:.3f}, "
        f"{rv['ci90_high']:.3f}] saved {row['saved_ms_pytree_minus_flat_vector']:.4f} ms; "
        f"pytree/flat_leaves {rl['ratio']:.3f} [{rl['ci90_low']:.3f}, {rl['ci90_high']:.3f}] "
        f"saved {row['saved_ms_pytree_minus_flat_leaves']:.4f} ms; "
        f"MDI {row['minimum_detectable_improvement']:.3f}"
    )
print("-" * 100)
print("  ALL ASSERTS PASSED (route agreement rtol 1e-10; grads finite and non-zero)")
for lane, verdict in phase2b_rule["per_lane"].items():
    print(
        f"  phase-2b rule [{lane}]: saved {verdict['saved_ms']:.4f} ms = "
        f"{verdict['fraction_of_fused']:.1%} of {verdict['fused_pytree_ms']:.4f} ms -> "
        f"{'GO' if verdict['go'] else 'no-go'} ({config_name}; RAL CPU decides)"
    )
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
