"""
JAX Profiling: PointSolver A100 Bottleneck Map (point-source GPU campaign, phase 0+1 lean)
==========================================================================================

One job on current ``main`` that measures what the CPU campaign's A100 no-regression rows
left unknown about the production single-source image-plane likelihood
(``FitPositionsImagePairAllSolved`` through ``PointSolver.for_grid``, 100 x 100 x 0.2",
precision 1e-3, MCS 20, the ``structured`` step-0 route), autolens_profiling#350.

Those rows (RAL jobs 350587 / 350637 / 357322 / 359102) showed the A100 call is
**launch-bound**: 0.83 ms scalar, 0.23 / 0.07 ms per likelihood under vmap-4 / 16, and flat
under FLOP-halving levers. This cell says *why* and *how far* -- the upper bound any phase-2
lever could reach -- so the go/no-go is a judgement against measured ceilings, not a guess.

Legs
----

1. **baseline** -- one lowering, two compilations: command buffers (CUDA graphs) ON (the XLA
   default, the production program) and OFF (``xla_gpu_enable_command_buffer=""``, the
   program every kernel can be named in). lower / compile / first-call, then interleaved
   ``--rounds`` x ``--calls`` warm calls over the fixed-seed stream (start rotated each
   round), per-call medians, bootstrap 90 % CI of OFF / ON. The fiducial gate: the ON
   program's log L at the prior-median vector equals ``7.743201200876806`` bit-exactly on the
   A100 (``...812`` on CPU); OFF must equal ON on every stream point.
2. **vmap** -- ``jax.vmap`` batches ``--vmap-batches`` (default 1/4/16/64/256; stops at the
   first OOM): ms per likelihood, throughput, compile s, XLA temp bytes, device
   ``memory_stats()``; gate ``max_abs_delta_vs_scalar`` <= ``VMAP_DELTA_TOL`` on every lane.
3. **trace** -- ``jax.profiler.trace`` of the command-buffers-OFF executable, scalar and
   vmap-``--trace-vmap``, parsed with ``scripts/misc/likelihood_breakdown/xla_attribution.py``
   and this cell's point-solver stage map (``_point_solver_stage_map.py``): kernel count per
   call, device-busy vs wall (the launch / host fraction), the inter-kernel gap distribution
   and a per-stage table with the lattice / active split. The ON program is traced too:
   CUPTI still emits its kernels (named by graph node), so its device-busy time is measured
   on the production program as well.
4. **fp32 what-if** -- run this cell a second time in a separate process with
   ``JAX_ENABLE_X64=0`` and ``--fp64-reference <the fp64 JSON>``. The PointSolver has NO
   mixed-precision switch (``al.Settings(use_mixed_precision)`` only reaches inversions), so
   this is whole-program fp32, labelled a what-if: timing, and |Δ log L|, finite image count
   and max position Δ against the fp64 run over the same stream. Evidence of headroom only,
   not a supported mode; its gates are reported, never required bit-exact.
5. **grad** -- reverse ``jax.value_and_grad`` and forward ``jax.jacfwd`` of the likelihood (5
   lens parameters, through the ``custom_jvp`` implicit rule): compile and warm timing,
   scalar and vmap-``--grad-vmap``. Throughput is quoted only beside a central
   finite-difference check, the ``autolens_workspace_test`` ``jax_grad/gradient.py``
   methodology: a fine-precision solver (1e-5) so the solver staircase is ~100x below
   production, a per-parameter step sweep (rel. steps 1e-4 .. 5e-3, floor 0.1) with the FD
   closest to autodiff used, ``|ad - fd| <= 1e-4 + 2 % max(|ad|, |fd|)``. A point where the
   fine solver's finite image count changes across its FD evaluations is a topology
   transition: flagged, never failed. ``autofit.jax.register_model(model)`` is load-bearing:
   without it ``jax.grad`` of this likelihood is silently all-zero.

Builders
--------

The model, dataset and solver are the control of ``solver_config_sweep.py``
(``_solved_model``, ``make_solver(control)`` = ``al.PointSolver.for_grid(grid=100x100@0.2,
pixel_scale_precision=0.001, magnification_threshold=0.1, neighbor_degree=1)``,
``loglike_factory``, ``_vector_stream(seed=314)``), copied rather than imported because that
script runs its sweep at import time. Nothing is patched: this cell measures the library as
it is.

Output
------

``results/breakdown/point_source_image/gpu_bottleneck_map_<config_name>.{json,png}``.
``laptop_*`` / ``local_*`` configs are load-inflated and recorded ``quotable: false``. No
top-level ``autolens_version``, so ``build_readme.py`` does not auto-table it.
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
_HERE = str(Path(__file__).resolve().parent)
for _p in (_MISC, str(_ROOT), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import math
import tempfile

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib
import numpy as np

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

import _point_solver_stage_map as stage_map  # noqa: E402
from likelihood_breakdown import xla_attribution  # noqa: E402
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

LEG_CHOICES = ("baseline", "vmap", "trace", "grad")

_cli = parse_profile_cli()
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--rounds", type=int, default=20)
_cell_parser.add_argument("--calls", type=int, default=20)
_cell_parser.add_argument("--n-stream", type=int, default=16, help="timing stream length")
_cell_parser.add_argument("--n-eval", type=int, default=64, help="stream points evaluated")
_cell_parser.add_argument("--warm", type=int, default=3)
_cell_parser.add_argument("--vmap-batches", default="1,4,16,64,256")
_cell_parser.add_argument("--vmap-rounds", type=int, default=10)
_cell_parser.add_argument("--vmap-calls", type=int, default=10)
_cell_parser.add_argument("--trace-calls", type=int, default=10)
_cell_parser.add_argument("--trace-vmap", type=int, default=16)
_cell_parser.add_argument("--steady-calls", type=int, default=50)
_cell_parser.add_argument("--grad-points", type=int, default=6, help="FD check points")
_cell_parser.add_argument("--grad-rounds", type=int, default=10)
_cell_parser.add_argument("--grad-calls", type=int, default=10)
_cell_parser.add_argument("--grad-vmap", type=int, default=16)
_cell_parser.add_argument("--seed", type=int, default=314)
_cell_parser.add_argument(
    "--legs", default=",".join(LEG_CHOICES), help=f"comma list of {','.join(LEG_CHOICES)}"
)
_cell_parser.add_argument(
    "--fp64-reference",
    default=None,
    help="fp64 JSON of this cell; with JAX_ENABLE_X64=0 the fp32 what-if compares against it",
)
_cell_parser.add_argument("--trace-dir", default=None, help="profiler output (default: temp)")
_cell_parser.add_argument(
    "--quick", action="store_true", help="laptop witness: tiny counts on every leg"
)
_args = _cli.parse_cell_args(_cell_parser)

QUICK = bool(_args.quick)
N_ROUNDS = 3 if QUICK else _args.rounds
N_CALLS = 3 if QUICK else _args.calls
N_STREAM = _args.n_stream
N_EVAL = 8 if QUICK else _args.n_eval
N_WARM = max(3, _args.warm)
VMAP_BATCHES = (1, 4) if QUICK else tuple(int(b) for b in _args.vmap_batches.split(",") if b)
N_VMAP_ROUNDS = 2 if QUICK else _args.vmap_rounds
N_VMAP_CALLS = 2 if QUICK else _args.vmap_calls
TRACE_CALLS = 3 if QUICK else _args.trace_calls
TRACE_VMAP = 4 if QUICK else _args.trace_vmap
STEADY_CALLS = 5 if QUICK else _args.steady_calls
N_GRAD_POINTS = 2 if QUICK else _args.grad_points
N_GRAD_ROUNDS = 2 if QUICK else _args.grad_rounds
N_GRAD_CALLS = 2 if QUICK else _args.grad_calls
GRAD_VMAP = 4 if QUICK else _args.grad_vmap
SEED = _args.seed
LEGS = [leg.strip() for leg in _args.legs.split(",") if leg.strip()]
_unknown = [leg for leg in LEGS if leg not in LEG_CHOICES]
if _unknown:
    raise SystemExit(f"--legs: unknown {_unknown}; choose from {LEG_CHOICES}")
if "baseline" not in LEGS:
    LEGS.insert(0, "baseline")  # every other leg is judged against the baseline
if min(N_ROUNDS, N_CALLS, N_STREAM, N_EVAL, TRACE_CALLS, STEADY_CALLS) < 1 or not VMAP_BATCHES:
    raise SystemExit("counts must be positive and --vmap-batches non-empty")

X64 = bool(jax.config.jax_enable_x64)
PRECISION = "fp64" if X64 else "fp32"
if not X64:
    # The fp32 what-if is timing + the delta table; the implicit-gradient FD check is an
    # fp64 certification and the trace attribution is precision-independent.
    LEGS = [leg for leg in LEGS if leg in ("baseline", "vmap")]

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autoarray.structures.triangles import array as _triangles_array  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402

BACKEND = jax.default_backend()
FIDUCIAL_SOLVED_LOG_L_BY_BACKEND = {"cpu": 7.743201200876812, "gpu": 7.743201200876806}
FIDUCIAL_SOLVED_LOG_L = FIDUCIAL_SOLVED_LOG_L_BY_BACKEND.get(
    BACKEND, FIDUCIAL_SOLVED_LOG_L_BY_BACKEND["cpu"]
)
PRODUCTION_PRECISION = 0.001
FINE_PRECISION = 1.0e-5
GRID_SHAPE = (100, 100)
GRID_PIXEL_SCALE = 0.2
VMAP_DELTA_TOL = 1.0e-9
FD_REL_STEPS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)
FD_ABS_FLOOR = 0.1
FD_RTOL = 2.0e-2
FD_ATOL = 1.0e-4
DISTINCT_TOL = 0.005
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
_COMMAND_BUFFER_OPTION = "xla_gpu_enable_command_buffer"

progress: list[str] = []


def note(msg):
    line = f"[{time.perf_counter() - _WALL_START:7.1f} s] {msg}"
    print(line, flush=True)
    progress.append(line)


# ---------------------------------------------------------------------------
# Builders: the solver_config_sweep.py control, byte for byte
# ---------------------------------------------------------------------------

_GRID = al.Grid2D.uniform(shape_native=GRID_SHAPE, pixel_scales=GRID_PIXEL_SCALE)


def make_solver(precision: float = PRODUCTION_PRECISION):
    return al.PointSolver.for_grid(
        grid=_GRID,
        pixel_scale_precision=precision,
        magnification_threshold=0.1,
        neighbor_degree=1,
    )


dataset_path = _ROOT / "dataset" / "point_source" / "simple"
auto_simulate_if_missing(
    dataset_path, dataset_type="point_source", instrument="simple", workspace_root=_ROOT
)
DATASET = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")


def _solved_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=af.Model(al.ps.PointSolved))
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


MODEL = _solved_model()
# Load-bearing: without it jax.grad of this likelihood is identically ZERO.
register_model_pytrees(MODEL)
PARAM_NAMES = [".".join(p) for p in MODEL.paths]


def _analysis(precision=PRODUCTION_PRECISION):
    return al.AnalysisPoint(
        dataset=DATASET,
        solver=make_solver(precision),
        fit_positions_cls=al.FitPositionsImagePairAllSolved,
        use_jax=True,
    )


def loglike_fn(precision=PRODUCTION_PRECISION):
    analysis = _analysis(precision)

    def log_likelihood(vector):
        instance = MODEL.instance_from_vector(vector=vector, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    return log_likelihood


def positions_fn(precision=PRODUCTION_PRECISION):
    """vector -> (log L, padded model positions)."""
    analysis = _analysis(precision)

    def fn(vector):
        instance = MODEL.instance_from_vector(vector=vector, xp=jnp)
        fit = analysis.fit_from(instance=instance)
        md = fit.positions.model_data
        return fit.log_likelihood, jnp.asarray(getattr(md, "array", md))

    return fn


def _vector_stream(model, n: int, seed: int) -> np.ndarray:
    """Fixed-seed physical vectors: entry 0 the prior medians, the rest U(0.25,0.75) draws."""
    rng = np.random.default_rng(seed)
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for _ in range(n - 1):
        unit = rng.uniform(0.25, 0.75, size=model.prior_count)
        vectors.append(np.asarray(model.vector_from_unit_vector(unit), dtype=float))
    return np.stack(vectors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        "generated_code_size_in_bytes",
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


def _device_memory():
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 -- a backend without memory stats is not an error
        return None
    keep = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "largest_alloc_size")
    return {k: int(stats[k]) for k in keep if k in stats} or None


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


def _mdi(samples_ms) -> float:
    """Minimum detectable improvement: the 90 % CI half-width of a same-program median ratio.

    Two halves of one route's own per-call samples, bootstrapped: what a null A/B of this
    program against itself resolves. A lever whose ceiling is below it cannot be measured.
    """
    s = np.asarray(samples_ms, dtype=float)
    half = s.size // 2
    if half < 4:
        return float("nan")
    r = _median_ratio(s[:half], s[half : 2 * half], BOOTSTRAP_SEED + 7)
    return float(max(abs(r["ci90_high"] - 1.0), abs(1.0 - r["ci90_low"])))


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(k in text for k in ("resource_exhausted", "out of memory", "oom", "allocat"))


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


def lower_and_compile(fn, example, *, label, both=True):
    """Fresh closure -> lower once -> compile ON (and OFF) -> first call. Returns (ex, rec)."""
    jax.clear_caches()
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(example)
    lower_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ex_on = lowered.compile()
    compile_on_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    block(ex_on(example))
    first_on_s = time.perf_counter() - t0
    rec = {
        "label": label,
        "lower_s": float(lower_s),
        "compile_s": float(compile_on_s),
        "first_call_s": float(first_on_s),
        "flops": _flops(ex_on),
        "memory_analysis": _memory_analysis(ex_on),
        "device_memory_after_first_call": _device_memory(),
    }
    ex_off = None
    if both:
        t0 = time.perf_counter()
        try:
            ex_off = lowered.compile(compiler_options={_COMMAND_BUFFER_OPTION: ""})
            rec["compile_off_s"] = float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            block(ex_off(example))
            rec["first_call_off_s"] = float(time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 -- CPU backends may reject the GPU option
            rec["compile_off_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            ex_off = None
    return (ex_on, ex_off), rec


def interleaved(executables: dict, args: list, rounds: int, calls: int, *, warm: int):
    """Round-robin rounds (start rotated), `calls` calls per route per round, varying args.

    Returns ({name: [seconds]}, {name: {arg_index: output}}).
    """
    names = list(executables)
    for name in names:
        for w in range(warm):
            block(executables[name](args[w % len(args)]))
    times = {n: [] for n in names}
    outputs = {n: {} for n in names}
    index = 0
    for r in range(rounds):
        order = names[r % len(names) :] + names[: r % len(names)]
        start = index
        for name in order:
            for c in range(calls):
                k = (start + c) % len(args)
                t0 = time.perf_counter()
                out = block(executables[name](args[k]))
                times[name].append(time.perf_counter() - t0)
                outputs[name].setdefault(k, out)
        index = start + calls
    return times, outputs


def steady_wall_ms(executable, args, n: int) -> float:
    block(executable(args[0]))
    t0 = time.perf_counter()
    for i in range(n):
        block(executable(args[i % len(args)]))
    return (time.perf_counter() - t0) / n * 1e3


def finite_positions(p):
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    return p[np.all(np.isfinite(p), axis=1)]


def distinct(p, tol=DISTINCT_TOL):
    out = []
    for q in p:
        if not any(np.hypot(*(q - r)) < tol for r in out):
            out.append(q)
    return np.asarray(out).reshape(-1, 2)


def max_matched_delta(a, b):
    """Max nearest-neighbour distance between two finite position sets (None if either empty)."""
    a, b = finite_positions(a), finite_positions(b)
    if a.size == 0 or b.size == 0:
        return None
    d_ab = np.min(np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)), axis=1)
    d_ba = np.min(np.hypot(*(b[:, None, :] - a[None, :, :]).transpose(2, 0, 1)), axis=1)
    return float(max(d_ab.max(), d_ba.max()))


# ---------------------------------------------------------------------------
# Leg 1: baseline (command buffers ON vs OFF) + stream evaluation
# ---------------------------------------------------------------------------

gates: dict = {}
stream = _vector_stream(MODEL, max(N_STREAM, N_EVAL), SEED)
arguments = [jnp.asarray(v) for v in stream[:N_STREAM]]
eval_arguments = [jnp.asarray(v) for v in stream[:N_EVAL]]
note(
    f"backend {BACKEND}, precision {PRECISION}, legs {LEGS}; MAX_CONTAINING_SIZE "
    f"{_triangles_array.MAX_CONTAINING_SIZE}, step-0 route "
    f"{getattr(_triangles_array, '_STEP0_CONTAINMENT', None)}, n_steps {make_solver().n_steps}"
)

(ex_on, ex_off), baseline_compile = lower_and_compile(
    loglike_fn(), arguments[0], label="scalar_log_likelihood"
)
note(
    f"baseline compile: lower {baseline_compile['lower_s']:.2f} s, compile ON "
    f"{baseline_compile['compile_s']:.2f} s, OFF {baseline_compile.get('compile_off_s')}"
)
routes = {"command_buffers_on": ex_on}
if ex_off is not None:
    routes["command_buffers_off"] = ex_off
b_times, b_out = interleaved(routes, arguments, N_ROUNDS, N_CALLS, warm=N_WARM)
baseline = {
    "compile": baseline_compile,
    "rows": {
        name: {"stats": _stats_ms(t), "per_call_ms": [x * 1e3 for x in t]}
        for name, t in b_times.items()
    },
}
baseline["mdi_same_program"] = _mdi(baseline["rows"]["command_buffers_on"]["per_call_ms"])
if ex_off is not None:
    baseline["off_over_on"] = _median_ratio(
        b_times["command_buffers_off"], b_times["command_buffers_on"], BOOTSTRAP_SEED
    )
    baseline["command_buffer_saving_ms"] = (
        baseline["rows"]["command_buffers_off"]["stats"]["median_ms"]
        - baseline["rows"]["command_buffers_on"]["stats"]["median_ms"]
    )
note(
    "baseline: "
    + ", ".join(f"{n} {r['stats']['median_ms']:.4f} ms" for n, r in baseline["rows"].items())
)

# Stream evaluation: log L + positions at every eval point (the fp32 what-if compares these).
(ex_pos, _), pos_compile = lower_and_compile(
    positions_fn(), eval_arguments[0], label="scalar_positions", both=False
)
stream_logl, stream_pos = [], []
for a in eval_arguments:
    lv, pv = block(ex_pos(a))
    stream_logl.append(float(lv))
    stream_pos.append(finite_positions(np.asarray(pv)).tolist())
stream_logl_on = [float(block(ex_on(a))) for a in eval_arguments]
stream_eval = {
    "seed": SEED,
    "vectors": stream[:N_EVAL].tolist(),
    "parameter_paths": PARAM_NAMES,
    "log_likelihood": stream_logl_on,
    "log_likelihood_positions_program": stream_logl,
    "finite_positions": stream_pos,
    "finite_image_count": [len(p) for p in stream_pos],
    "distinct_image_count": [len(distinct(np.asarray(p).reshape(-1, 2))) for p in stream_pos],
    "positions_compile": pos_compile,
}

fid = stream_logl_on[0]
if X64:
    gates["fiducial_bit_exact"] = {
        "value": fid,
        "repr": repr(fid),
        "expected": FIDUCIAL_SOLVED_LOG_L,
        "pass": fid == FIDUCIAL_SOLVED_LOG_L,
    }
if ex_off is not None:
    off_vals = [float(block(ex_off(a))) for a in eval_arguments]
    diffs = [a != b and not (np.isnan(a) and np.isnan(b)) for a, b in zip(off_vals, stream_logl_on)]
    gates["command_buffers_off_bit_identical"] = {
        "n_points": len(off_vals),
        "n_differ": int(sum(diffs)),
        "pass": not any(diffs),
    }
gates["stream_finite"] = {
    "n_points": len(stream_logl_on),
    "n_non_finite": int(sum(not np.isfinite(v) for v in stream_logl_on)),
    "pass": all(np.isfinite(v) for v in stream_logl_on),
}
note(f"fiducial {fid!r} (expected {FIDUCIAL_SOLVED_LOG_L!r}); gates {gates}")


# ---------------------------------------------------------------------------
# Leg 2: vmap scaling
# ---------------------------------------------------------------------------

vmap_block = None
if "vmap" in LEGS:
    n_vstream = max(VMAP_BATCHES)
    vstream = _vector_stream(MODEL, n_vstream, SEED + 1)
    (ex_vs, _), _rec = lower_and_compile(
        loglike_fn(), jnp.asarray(vstream[0]), label="vmap_scalar_ref", both=False
    )
    scalar_ref = np.asarray([float(block(ex_vs(jnp.asarray(v)))) for v in vstream])
    vmap_block = {"batches": list(VMAP_BATCHES), "rows": {}, "stopped_at": None}
    for b in VMAP_BATCHES:
        n_batches = max(2, min(8, n_vstream // b))
        idx = [np.arange(j * b, j * b + b) % n_vstream for j in range(n_batches)]
        batch_args = [jnp.asarray(vstream[i]) for i in idx]
        try:
            (ex_b, _), rec = lower_and_compile(
                jax.vmap(loglike_fn()), batch_args[0], label=f"vmap{b}", both=False
            )
            t, outs = interleaved(
                {"vmap": ex_b}, batch_args, N_VMAP_ROUNDS, N_VMAP_CALLS, warm=N_WARM
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            vmap_block["stopped_at"] = {"batch": b, "error": f"{type(exc).__name__}: {exc}"[:500]}
            note(f"vmap{b}: OOM -> stop ({type(exc).__name__})")
            break
        st = _stats_ms(t["vmap"])
        deltas = [
            float(np.nanmax(np.abs(np.asarray(outs["vmap"][k], dtype=float) - scalar_ref[idx[k]])))
            for k in sorted(outs["vmap"])
        ]
        vmap_block["rows"][str(b)] = {
            "batch": b,
            "stats_per_batch": st,
            "median_ms_per_likelihood": st["median_ms"] / b,
            "throughput_likelihoods_per_s": b / (st["median_ms"] / 1e3),
            "compile": rec,
            "device_memory_after_timing": _device_memory(),
            "max_abs_delta_vs_scalar": max(deltas),
        }
        note(
            f"vmap{b}: {st['median_ms']:.4f} ms/batch = {st['median_ms'] / b:.5f} ms/L, "
            f"compile {rec['compile_s']:.1f} s, delta {max(deltas):.2e}"
        )
    for row in vmap_block["rows"].values():
        row["speedup_per_likelihood_vs_vmap1"] = (
            vmap_block["rows"]["1"]["median_ms_per_likelihood"] / row["median_ms_per_likelihood"]
            if "1" in vmap_block["rows"]
            else None
        )
    vmap_block["largest_batch_fitting"] = max((int(b) for b in vmap_block["rows"]), default=None)
    gates["vmap_matches_scalar"] = {
        "tol": VMAP_DELTA_TOL,
        "max_abs_delta": {b: r["max_abs_delta_vs_scalar"] for b, r in vmap_block["rows"].items()},
        "pass": all(
            (r["max_abs_delta_vs_scalar"] <= VMAP_DELTA_TOL) if X64 else True
            for r in vmap_block["rows"].values()
        ),
    }


# ---------------------------------------------------------------------------
# Leg 3: device trace + stage map
# ---------------------------------------------------------------------------


def _hlo_launch_census(index, rules, hlo_text: str) -> dict:
    """Static census of the optimized HLO: ENTRY-computation instructions per stage.

    Every instruction of the ENTRY computation that is not pure bookkeeping is one
    executable unit (a fusion, a custom call, a sort, a reduce ...): a static estimate of
    the kernel launches per call, available on every backend (the CPU laptop witness has
    no device timeline). The program has no ``while`` / ``conditional`` (the solver's step
    loop is unrolled), so ENTRY is the whole schedule.
    """
    import re

    match = re.search(r"^ENTRY\s+%?(?P<name>[\w.\-$]+)", hlo_text, re.M)
    entry = match.group("name") if match else None
    loops = len(re.findall(r"\b(?:while|conditional)\(", hlo_text))
    skip = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast", "after-all"}
    per_stage: dict[str, int] = {}
    opcodes: dict[str, int] = {}
    other_sources: dict[str, int] = {}
    mixed_sets: dict[str, int] = {}
    for inst in index.values():
        if inst.computation != entry or inst.opcode in skip:
            continue
        label = stage_map.classify(inst, index, rules)
        per_stage[label] = per_stage.get(label, 0) + 1
        opcodes[inst.opcode] = opcodes.get(inst.opcode, 0) + 1
        if label == xla_attribution.OTHER:
            key = f"{inst.opcode} @ {inst.source or inst.op_name or '<no source>'}"
            other_sources[key] = other_sources.get(key, 0) + 1
        elif label == xla_attribution.MIXED_FUSION:
            _, members = xla_attribution.stages_of_instruction(inst, index, rules)
            key = " + ".join(sorted(members))
            mixed_sets[key] = mixed_sets.get(key, 0) + 1
    return {
        "entry_computation": entry,
        "while_or_conditional_ops": loops,
        "top_level_instructions": sum(per_stage.values()),
        "per_stage": dict(sorted(per_stage.items(), key=lambda kv: -kv[1])),
        "opcodes": dict(sorted(opcodes.items(), key=lambda kv: -kv[1])),
        "other_sources_top": dict(sorted(other_sources.items(), key=lambda kv: -kv[1])[:25]),
        "mixed_fusion_sets_top": dict(sorted(mixed_sets.items(), key=lambda kv: -kv[1])[:15]),
        "instructions_total": len(index),
        "instructions_resolved_to_source": sum(1 for i in index.values() if i.frames),
    }


def _cpu_hlo_events(log_dir) -> list[dict]:
    """CPU-backend stand-in for ``device_events``: host-plane events that carry ``hlo_op``."""
    profile = jax.profiler.ProfileData.from_file(xla_attribution.find_xplane(log_dir))
    events: list[dict] = []
    for plane in profile.planes:
        if "/host:CPU" not in plane.name:
            continue
        for line in plane.lines:
            for event in line.events:
                stats = dict(event.stats)
                if "hlo_op" not in stats:
                    continue
                events.append(
                    {
                        "plane": plane.name,
                        "stream": line.name,
                        "name": event.name,
                        "start_ns": int(event.start_ns),
                        "dur_ns": int(event.duration_ns),
                        "hlo_op": stats.get("hlo_op"),
                        "hlo_module": stats.get("hlo_module"),
                        "program_id": stats.get("program_id"),
                    }
                )
    events.sort(key=lambda e: e["start_ns"])
    return events


trace_block = None
if "trace" in LEGS:
    trace_root = Path(_args.trace_dir) if _args.trace_dir else Path(tempfile.mkdtemp())
    trace_root.mkdir(parents=True, exist_ok=True)
    ranges = stage_map.library_ranges()
    rules = stage_map.build_stage_map(ranges)
    trace_block = {"library_ranges": ranges, "trace_dir": trace_root.name, "programs": {}}
    tb = TRACE_VMAP
    tstream = _vector_stream(MODEL, max(tb, N_STREAM), SEED + 2)
    programs = [
        ("scalar", loglike_fn(), [jnp.asarray(v) for v in tstream[:N_STREAM]], 1),
        (
            f"vmap{tb}",
            jax.vmap(loglike_fn()),
            [jnp.asarray(np.roll(tstream, -j, axis=0)[:tb]) for j in range(4)],
            tb,
        ),
    ]
    for label, fn, targs, lanes in programs:
        (t_on, t_off), rec = lower_and_compile(fn, targs[0], label=f"trace_{label}")
        traced_ex = t_off if t_off is not None else t_on
        wall_on = steady_wall_ms(t_on, targs, STEADY_CALLS)
        wall_off = steady_wall_ms(t_off, targs, STEADY_CALLS) if t_off is not None else None
        hlo_text = traced_ex.as_text()
        proto = traced_ex.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()
        (trace_root / f"{label}_hlo_optimized.txt").write_text(hlo_text)
        index = xla_attribution.hlo_index(
            hlo_text, xla_attribution.StackFrameIndex.from_module_proto(proto)
        )
        census = _hlo_launch_census(index, rules, hlo_text)
        main_dir = trace_root / f"{label}_traced_off"
        # The clock starts INSIDE the profiler context (fixed_light_trace.py precedent), and
        # no warm-up call is made inside it, so the trace holds exactly TRACE_CALLS calls.
        with jax.profiler.trace(
            str(main_dir), create_perfetto_link=False, create_perfetto_trace=False
        ):
            t0 = time.perf_counter()
            for i in range(TRACE_CALLS):
                block(traced_ex(targs[i % len(targs)]))
            traced_wall = (time.perf_counter() - t0) / TRACE_CALLS * 1e3
        untraced_wall = steady_wall_ms(traced_ex, targs, STEADY_CALLS)
        try:
            events = xla_attribution.device_events(main_dir)
        except FileNotFoundError as exc:
            events = []
            rec["trace_error"] = str(exc)
        timeline_source = "device plane (xla_attribution.device_events)"
        if not events and BACKEND == "cpu":
            events = _cpu_hlo_events(main_dir)
            timeline_source = (
                "host:CPU events carrying hlo_op (laptop witness: multi-threaded thunks, "
                "overlapping; exercises the join and the stage map, not a device timeline)"
            )
        stages = stage_map.stage_table(events, index, calls=TRACE_CALLS, stage_map=rules)
        try:
            attribution = xla_attribution.attribute(
                events,
                index,
                wall_ms=traced_wall,
                calls=TRACE_CALLS,
                stage_map=rules,
                untraced_wall_ms=untraced_wall,
            )
        except TypeError as exc:
            # xla_attribution.idle_gaps compares event dicts on an exact (start, end) tie
            # (seen on the multi-threaded CPU witness); keep the cell's own stage table and
            # its busy / span, and record why the shared attribution block is missing.
            attribution = {
                "error": f"{type(exc).__name__}: {exc}",
                "device_busy_ms": xla_attribution._union_busy_ns(events) / 1e6 / TRACE_CALLS,
                "unjoined_ms": stages["per_stage"]
                .get(xla_attribution.UNJOINED, {})
                .get("median_ms", 0.0),
                "unjoined_pct": 100.0
                * stages["per_stage"].get(xla_attribution.UNJOINED, {}).get("median_ms", 0.0)
                / traced_wall,
            }
        # The production (command buffers ON) program: CUPTI still emits its kernels, named
        # by graph node, so its device-busy time is measured even though stages are not.
        on_probe = None
        if t_off is not None:
            probe_dir = trace_root / f"{label}_probe_on"
            with jax.profiler.trace(
                str(probe_dir), create_perfetto_link=False, create_perfetto_trace=False
            ):
                t0 = time.perf_counter()
                for i in range(TRACE_CALLS):
                    block(t_on(targs[i % len(targs)]))
                probe_wall = (time.perf_counter() - t0) / TRACE_CALLS * 1e3
            try:
                p_events = xla_attribution.device_events(probe_dir)
            except FileNotFoundError:
                p_events = []
            p_busy = xla_attribution._union_busy_ns(p_events) / 1e6 / TRACE_CALLS
            p_span = (
                (
                    max(e["start_ns"] + e["dur_ns"] for e in p_events)
                    - min(e["start_ns"] for e in p_events)
                )
                / 1e6
                / TRACE_CALLS
                if p_events
                else 0.0
            )
            on_probe = {
                "events_per_call": len(p_events) / TRACE_CALLS,
                "distinct_hlo_op": len({e["hlo_op"] for e in p_events}),
                "device_busy_ms_per_call": p_busy,
                "device_span_ms_per_call_total_over_calls": p_span,
                "traced_wall_ms": probe_wall,
                "untraced_wall_ms": wall_on,
                "busy_over_untraced_wall_pct": 100.0 * p_busy / wall_on if wall_on else None,
            }
        wall_ref = untraced_wall if untraced_wall else traced_wall
        busy = attribution["device_busy_ms"]
        trace_block["programs"][label] = {
            "lanes": lanes,
            "timeline_source": timeline_source,
            "compile": rec,
            "wall_command_buffers_on_ms": wall_on,
            "wall_command_buffers_off_ms": wall_off,
            "traced_wall_ms": traced_wall,
            "untraced_wall_ms": untraced_wall,
            "events_total": len(events),
            "kernels_per_call": stages["kernels_per_call"],
            "device_busy_ms_per_call": busy,
            "device_busy_over_untraced_wall_pct": 100.0 * busy / wall_ref if wall_ref else None,
            "launch_and_host_fraction_pct": 100.0 * (1.0 - busy / wall_ref) if wall_ref else None,
            "stages": stages,
            "attribution": attribution,
            "hlo_launch_census": census,
            "command_buffers_on_probe": on_probe,
        }
        note(
            f"trace {label}: {stages['kernels_per_call']} kernels/call, busy {busy:.4f} ms vs "
            f"wall OFF {untraced_wall:.4f} / ON {wall_on:.4f} ms; top stages "
            + ", ".join(
                f"{k} {v['median_ms']:.4f}"
                for k, v in sorted(stages["per_stage"].items(), key=lambda kv: -kv[1]["median_ms"])[
                    :4
                ]
            )
        )
    scalar_prog = trace_block["programs"].get("scalar", {})
    gates["trace_join"] = {
        "unjoined_ms": (scalar_prog.get("attribution") or {}).get("unjoined_ms"),
        "events": scalar_prog.get("events_total"),
        "note": "reported; on CPU there is no device timeline and events == 0",
        "pass": True
        if BACKEND != "gpu"
        else bool(scalar_prog.get("events_total"))
        and (scalar_prog["attribution"]["unjoined_pct"] or 0.0) < 5.0,
    }


# ---------------------------------------------------------------------------
# Leg 5: gradients (reverse + forward) + FD certification
# ---------------------------------------------------------------------------

grad_block = None
if "grad" in LEGS:
    grad_block = {}

    def rev_fn():
        return jax.value_and_grad(loglike_fn())

    def fwd_fn():
        f = loglike_fn()
        return lambda v: (f(v), jax.jacfwd(f)(v))

    (ex_rev, _), rec_rev = lower_and_compile(rev_fn(), arguments[0], label="grad_rev", both=False)
    (ex_fwd, _), rec_fwd = lower_and_compile(fwd_fn(), arguments[0], label="grad_fwd", both=False)
    g_times, g_out = interleaved(
        {"primal": ex_on, "reverse": ex_rev, "forward": ex_fwd},
        arguments,
        N_GRAD_ROUNDS,
        N_GRAD_CALLS,
        warm=N_WARM,
    )
    grad_block["scalar"] = {
        name: {"stats": _stats_ms(t), "over_primal": _median_ratio(t, g_times["primal"], 11)}
        for name, t in g_times.items()
    }
    grad_block["scalar"]["reverse"]["compile"] = rec_rev
    grad_block["scalar"]["forward"]["compile"] = rec_fwd
    rev_vals = {k: np.asarray(v[1], dtype=float) for k, v in g_out["reverse"].items()}
    fwd_vals = {k: np.asarray(v[1], dtype=float) for k, v in g_out["forward"].items()}
    # Forward vs reverse on every timing-stream point, with finiteness recorded: a NaN in
    # either mode must be visible here, not only as a NaN max (RAL jobs 366913 / 366915
    # reported this block's max as null -- forward mode was NaN).
    grad_block["reverse_vs_forward"] = {
        str(k): {
            "reverse": rev_vals[k].tolist(),
            "forward": fwd_vals[k].tolist(),
            "reverse_finite": bool(np.isfinite(rev_vals[k]).all()),
            "forward_finite": bool(np.isfinite(fwd_vals[k]).all()),
            "max_rel": float(
                np.max(np.abs(rev_vals[k] - fwd_vals[k]) / np.maximum(np.abs(rev_vals[k]), 1e-300))
            ),
        }
        for k in sorted(rev_vals)
        if k in fwd_vals
    }
    _rf = grad_block["reverse_vs_forward"].values()
    grad_block["reverse_vs_forward_max_rel"] = float(max(r["max_rel"] for r in _rf))
    gates["grad_reverse_finite"] = {
        "n_points": len(_rf),
        "n_non_finite": sum(not r["reverse_finite"] for r in _rf),
        "pass": all(r["reverse_finite"] for r in _rf),
    }
    gates["grad_forward_finite"] = {
        "n_points": len(_rf),
        "n_non_finite": sum(not r["forward_finite"] for r in _rf),
        "note": (
            "jax.jacfwd of the solved image-plane likelihood -- the mode AnalysisPoint declares "
            "(gradient_mode = 'forward', PyAutoLens#752) for gradient searches"
        ),
        "pass": all(r["forward_finite"] for r in _rf),
    }
    # vmap of each
    gb = GRAD_VMAP
    gbatch = [jnp.asarray(np.roll(stream[:N_STREAM], -j, axis=0)[:gb]) for j in range(4)]
    if gb > N_STREAM:
        gbatch = [jnp.asarray(_vector_stream(MODEL, gb, SEED + 3 + j)) for j in range(4)]
    (ex_vp, _), rec_vp = lower_and_compile(
        jax.vmap(loglike_fn()), gbatch[0], label=f"grad_vmap{gb}_primal", both=False
    )
    (ex_vr, _), rec_vr = lower_and_compile(
        jax.vmap(rev_fn()), gbatch[0], label=f"grad_vmap{gb}_rev", both=False
    )
    (ex_vf, _), rec_vf = lower_and_compile(
        jax.vmap(fwd_fn()), gbatch[0], label=f"grad_vmap{gb}_fwd", both=False
    )
    v_times, _ = interleaved(
        {"primal": ex_vp, "reverse": ex_vr, "forward": ex_vf},
        gbatch,
        N_GRAD_ROUNDS,
        N_GRAD_CALLS,
        warm=N_WARM,
    )
    grad_block[f"vmap{gb}"] = {
        name: {
            "stats_per_batch": _stats_ms(t),
            "median_ms_per_likelihood": float(np.median(t)) * 1e3 / gb,
            "over_primal": _median_ratio(t, v_times["primal"], 13),
        }
        for name, t in v_times.items()
    }
    grad_block[f"vmap{gb}"]["primal"]["compile"] = rec_vp
    grad_block[f"vmap{gb}"]["reverse"]["compile"] = rec_vr
    grad_block[f"vmap{gb}"]["forward"]["compile"] = rec_vf
    note(
        "grad scalar: "
        + ", ".join(
            f"{n} {r['stats']['median_ms']:.4f} ms" for n, r in grad_block["scalar"].items()
        )
        + f"; vmap{gb}: "
        + ", ".join(
            f"{n} {r['median_ms_per_likelihood']:.5f} ms/L"
            for n, r in grad_block[f"vmap{gb}"].items()
        )
    )

    # --- FD certification on the fine-precision solver --------------------------------
    (ex_fine_pos, _), rec_fine = lower_and_compile(
        positions_fn(FINE_PRECISION), arguments[0], label="fd_fine_positions", both=False
    )
    (ex_fine_grad, _), _rec = lower_and_compile(
        jax.grad(loglike_fn(FINE_PRECISION)), arguments[0], label="fd_fine_grad", both=False
    )
    (ex_fine_fwd, _), _rec = lower_and_compile(
        jax.jacfwd(loglike_fn(FINE_PRECISION)), arguments[0], label="fd_fine_fwd", both=False
    )

    def fine_eval(x):
        lv, pv = block(ex_fine_pos(jnp.asarray(x)))
        return float(lv), finite_positions(np.asarray(pv))

    def set_shift(a, b):
        """Max nearest-neighbour move of the solved-image set (None if the count changed)."""
        return max_matched_delta(a, b) if len(a) == len(b) else None

    # Stream point 0 is the prior-median vector: lens centre (0, 0) and ell_comps_1 = 0, an
    # axis-aligned symmetric lens. The workspace_test certification deliberately offsets its
    # base point off that symmetry (U(2e-4, 6e-4)); here point 0 is run and REPORTED as a
    # diagnostic but not gated, and the gated points are stream points 1..N (U(0.25,0.75)
    # prior-core draws, generic).
    points = []
    for p in range(0, min(N_GRAD_POINTS + 1, N_STREAM)):
        x = np.asarray(stream[p], dtype=float)
        ad_prod = np.asarray(rev_vals.get(p, block(ex_rev(jnp.asarray(x)))[1]), dtype=float)
        ad_fine = np.asarray(block(ex_fine_grad(jnp.asarray(x))), dtype=float)
        _, base_set = fine_eval(x)
        n_base = len(base_set)
        fd_all = np.zeros((len(FD_REL_STEPS), x.size))
        count_changed = np.zeros((len(FD_REL_STEPS), x.size), dtype=bool)
        counts_pm = np.zeros((len(FD_REL_STEPS), x.size, 2), dtype=int)
        shift_pm = np.full((len(FD_REL_STEPS), x.size, 2), np.nan)
        for s, rel in enumerate(FD_REL_STEPS):
            for i in range(x.size):
                h = rel * max(abs(x[i]), FD_ABS_FLOOR)
                xp_, xm_ = x.copy(), x.copy()
                xp_[i] += h
                xm_[i] -= h
                fp, set_p = fine_eval(xp_)
                fm, set_m = fine_eval(xm_)
                fd_all[s, i] = (fp - fm) / (2.0 * h)
                counts_pm[s, i] = (len(set_m), len(set_p))
                count_changed[s, i] = len(set_p) != n_base or len(set_m) != n_base
                for j, other in enumerate((set_m, set_p)):
                    d = set_shift(other, base_set)
                    shift_pm[s, i, j] = np.nan if d is None else d
        best = np.argmin(np.abs(fd_all - ad_fine[None, :]), axis=0)
        fd = fd_all[best, np.arange(x.size)]
        abs_err = np.abs(ad_fine - fd)
        denom = np.maximum(np.abs(ad_fine), np.abs(fd))
        rel_err = np.divide(abs_err, denom, out=np.zeros_like(abs_err), where=denom > 0)
        tol = FD_ATOL + FD_RTOL * denom
        strict = abs_err <= tol
        # DIAGNOSTIC ONLY (never the gate): the FD side carries the solver staircase as
        # noise whose absolute size is set by stair height / step, so the spread of the FD
        # sweep across steps is recorded per component, with a vector criterion. A
        # classification built from it was briefly the gate (commit 8c12e57, run 366915);
        # it was reverted: a smooth point whose FD disagrees at every step stays FAILING.
        noise_floor = np.std(fd_all, axis=0)
        at_noise_floor = ~strict & (abs_err <= noise_floor)
        vector_rel = float(np.linalg.norm(ad_fine - fd) / max(np.linalg.norm(fd), 1e-300))
        rel_err_all_steps = np.abs(fd_all - ad_fine[None, :]) / np.maximum(
            np.maximum(np.abs(fd_all), np.abs(ad_fine)[None, :]), 1e-300
        )
        # A topology transition: the finite image count changes between x-h and x+h at ANY
        # step of the sweep for that component.
        transition = bool(count_changed.any())
        ad_fine_fwd = np.asarray(block(ex_fine_fwd(jnp.asarray(x))), dtype=float)
        prod_rel = np.abs(ad_prod - ad_fine) / np.maximum(np.abs(ad_fine), 1e-300)
        points.append(
            {
                "point": p,
                "vector": x.tolist(),
                "fine_finite_image_count": n_base,
                "ad_fine": ad_fine.tolist(),
                "ad_production": ad_prod.tolist(),
                "fd_fine_used": fd.tolist(),
                "fd_step_used": [FD_REL_STEPS[b] for b in best],
                "fd_sweep": fd_all.tolist(),
                "rel_err_ad_vs_fd": rel_err.tolist(),
                "rel_err_ad_vs_fd_every_step": rel_err_all_steps.tolist(),
                "finite_count_minus_plus_every_step": counts_pm.tolist(),
                "solved_set_shift_minus_plus_every_step_arcsec": shift_pm.tolist(),
                "ad_fine_forward": ad_fine_fwd.tolist(),
                "ad_fine_forward_finite": bool(np.isfinite(ad_fine_fwd).all()),
                "strict_componentwise_pass_per_param": strict.tolist(),
                "strict_componentwise_pass": bool(strict.all()),
                "pass_per_param": strict.tolist(),
                "pass": bool(strict.all()),
                "diagnostic_noise_floor_classification": {
                    "fd_sweep_std_per_param": noise_floor.tolist(),
                    "within_sweep_std_per_param": at_noise_floor.tolist(),
                    "vector_rel_err": vector_rel,
                    "would_pass": bool((strict | at_noise_floor).all() and vector_rel <= FD_RTOL),
                    "note": "diagnostic only; NOT the gate (see fd_check.gate_history)",
                },
                "topology_transition": transition,
                "gated": p != 0,
                "role": "symmetric prior-median diagnostic (not gated)" if p == 0 else "gated",
                "production_vs_fine_ad_rel": prod_rel.tolist(),
                "all_finite_nonzero": bool(np.isfinite(ad_prod).all() and np.any(ad_prod != 0)),
            }
        )
        note(
            f"FD point {p}: max rel err {rel_err.max():.2e} strict pass {bool(strict.all())} "
            f"(vector rel {vector_rel:.2e}; fine fwd finite {bool(np.isfinite(ad_fine_fwd).all())}) "
            f"transition {transition}; production-vs-fine AD max rel {prod_rel.max():.2e}"
        )
    smooth = [pt for pt in points if pt["gated"] and not pt["topology_transition"]]
    grad_block["fd_check"] = {
        "method": (
            "autolens_workspace_test scripts/point_source/jax_grad/gradient.py: fine-precision "
            f"solver ({FINE_PRECISION}), rel steps {FD_REL_STEPS} x max(|x|, {FD_ABS_FLOOR}), FD "
            f"closest to AD used, |ad-fd| <= {FD_ATOL} + {FD_RTOL} max(|ad|,|fd|) per component "
            "(the gate). AD = reverse mode on the fine solver."
        ),
        "gate_history": (
            "366913: strict gate, failed points 3/4. 8c12e57 / 366915: gate briefly widened to "
            "a noise-floor + vector classification after that failure. Reverted: a gate changed "
            "after a failure must be justified from the data, and a smooth point whose FD "
            "disagrees at every step is a finding, not noise to tune away. The widened "
            "classification is kept per point as diagnostic_noise_floor_classification."
        ),
        "fine_compile": rec_fine,
        "parameter_paths": PARAM_NAMES,
        "points": points,
        "n_points": len(points),
        "n_gated": sum(pt["gated"] for pt in points),
        "n_topology_transition": sum(pt["gated"] and pt["topology_transition"] for pt in points),
        "n_smooth_pass": sum(pt["pass"] for pt in smooth),
        "n_smooth_strict_pass": sum(pt["strict_componentwise_pass"] for pt in smooth),
        "failing_smooth": [
            {
                "point": pt["point"],
                "params": [PARAM_NAMES[i] for i, ok in enumerate(pt["pass_per_param"]) if not ok],
            }
            for pt in smooth
            if not pt["pass"]
        ],
    }
    gates["grad_finite_nonzero"] = {"pass": all(pt["all_finite_nonzero"] for pt in points)}
    gates["grad_fd_agreement"] = {
        "n_smooth": len(smooth),
        "n_smooth_pass": grad_block["fd_check"]["n_smooth_pass"],
        "flagged_transitions": [
            pt["point"] for pt in points if pt["gated"] and pt["topology_transition"]
        ],
        "symmetric_point_0_pass": next((pt["pass"] for pt in points if pt["point"] == 0), None),
        "pass": bool(smooth) and all(pt["pass"] for pt in smooth),
    }


# ---------------------------------------------------------------------------
# Leg 4: fp32 what-if (this process is the fp32 one; compare with the fp64 JSON)
# ---------------------------------------------------------------------------

fp32_whatif = None
if not X64:
    ref_path = Path(_args.fp64_reference) if _args.fp64_reference else None
    fp32_whatif = {
        "label": (
            "WHAT-IF: whole-program JAX_ENABLE_X64=0 in a separate process. The PointSolver has "
            "no mixed-precision switch; this is headroom evidence, not a supported mode."
        ),
        "fp64_reference": str(ref_path) if ref_path else None,
    }
    if ref_path is not None and ref_path.exists():
        ref = json.loads(ref_path.read_text())
        ref_eval = ref["stream_eval"]
        n = min(len(ref_eval["log_likelihood"]), len(stream_logl_on))
        same_vectors = np.allclose(np.asarray(ref_eval["vectors"][:n]), stream[:n], rtol=0, atol=0)
        dlogl = [abs(stream_logl_on[i] - ref_eval["log_likelihood"][i]) for i in range(n)]
        count_match = [
            stream_eval["finite_image_count"][i] == ref_eval["finite_image_count"][i]
            for i in range(n)
        ]
        pos_delta = [
            max_matched_delta(stream_eval["finite_positions"][i], ref_eval["finite_positions"][i])
            for i in range(n)
        ]
        finite_pd = [d for d in pos_delta if d is not None]
        ref_on = ref["baseline"]["rows"]["command_buffers_on"]
        fp32_whatif.update(
            {
                "n_points": n,
                "same_stream_vectors": bool(same_vectors),
                "abs_delta_log_likelihood": {
                    "max": float(np.max(dlogl)),
                    "median": float(np.median(dlogl)),
                    "per_point": dlogl,
                },
                "finite_image_count_agreement": float(np.mean(count_match)),
                "n_image_count_mismatch": int(n - sum(count_match)),
                "max_position_delta_arcsec": float(max(finite_pd)) if finite_pd else None,
                "median_position_delta_arcsec": float(np.median(finite_pd)) if finite_pd else None,
                "fp64_over_fp32_scalar_on": _median_ratio(
                    np.asarray(ref_on["per_call_ms"]),
                    np.asarray(baseline["rows"]["command_buffers_on"]["per_call_ms"]),
                    BOOTSTRAP_SEED + 3,
                ),
                "fp64_scalar_on_median_ms": ref_on["stats"]["median_ms"],
                "fp32_scalar_on_median_ms": baseline["rows"]["command_buffers_on"]["stats"][
                    "median_ms"
                ],
                "fp64_reference_job": (ref.get("device", {}).get("provenance") or {}).get("slurm"),
            }
        )
        if vmap_block and ref.get("vmap"):
            fp32_whatif["vmap_fp64_over_fp32_ms_per_likelihood"] = {
                b: ref["vmap"]["rows"][b]["median_ms_per_likelihood"]
                / vmap_block["rows"][b]["median_ms_per_likelihood"]
                for b in vmap_block["rows"]
                if b in ref["vmap"]["rows"]
            }
        note(
            f"fp32 what-if: max |dlogL| {fp32_whatif['abs_delta_log_likelihood']['max']:.3e}, "
            f"image-count agreement {fp32_whatif['finite_image_count_agreement']:.3f}, max pos "
            f"delta {fp32_whatif['max_position_delta_arcsec']}, fp64/fp32 "
            f"{fp32_whatif['fp64_over_fp32_scalar_on']['ratio']:.3f}"
        )
    else:
        fp32_whatif["note"] = "no --fp64-reference JSON found; deltas not computed"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

CELL = "gpu_bottleneck_map"
config_name = _cli.config_name or f"local_{BACKEND}_{PRECISION}"
device = device_info_dict()
if device.get("jax_compilation_cache_dir"):
    device["jax_compilation_cache_dir"] = (
        "<scrubbed>/" + Path(device["jax_compilation_cache_dir"]).name
    )

required = ["fiducial_bit_exact", "command_buffers_off_bit_identical", "stream_finite"]
required += ["vmap_matches_scalar", "trace_join", "grad_finite_nonzero", "grad_fd_agreement"]
required += ["grad_reverse_finite", "grad_forward_finite"]
if not X64:
    # fp32 what-if: its gates are REPORTED, never required bit-exact; the run is valid when
    # it executed and every stream point is finite.
    required = ["stream_finite"]
if BACKEND == "gpu":
    gates["backend_gpu"] = {"backend": BACKEND, "pass": True}
    required.append("backend_gpu")
all_gates_pass = bool(all(gates[g]["pass"] for g in required if g in gates))

on_ms = baseline["rows"]["command_buffers_on"]["stats"]["median_ms"]
headline = {
    "scalar_command_buffers_on_ms": on_ms,
    "scalar_command_buffers_off_ms": (baseline["rows"].get("command_buffers_off") or {})
    .get("stats", {})
    .get("median_ms"),
    "mdi_same_program": baseline["mdi_same_program"],
}
if vmap_block:
    headline["vmap_ms_per_likelihood"] = {
        b: r["median_ms_per_likelihood"] for b, r in vmap_block["rows"].items()
    }
    headline["largest_batch_fitting"] = vmap_block["largest_batch_fitting"]
if trace_block:
    for label, prog in trace_block["programs"].items():
        headline[f"trace_{label}"] = {
            "kernels_per_call": prog["kernels_per_call"],
            "device_busy_ms": prog["device_busy_ms_per_call"],
            "untraced_wall_off_ms": prog["untraced_wall_ms"],
            "wall_on_ms": prog["wall_command_buffers_on_ms"],
            "device_busy_over_wall_off_pct": prog["device_busy_over_untraced_wall_pct"],
            "on_program_busy_over_wall_pct": (prog["command_buffers_on_probe"] or {}).get(
                "busy_over_untraced_wall_pct"
            ),
        }
if grad_block:
    headline["grad_scalar_ms"] = {
        n: r["stats"]["median_ms"] for n, r in grad_block["scalar"].items()
    }
if fp32_whatif and "abs_delta_log_likelihood" in fp32_whatif:
    headline["fp32_whatif"] = {
        "max_abs_dlogl": fp32_whatif["abs_delta_log_likelihood"]["max"],
        "fp64_over_fp32": fp32_whatif["fp64_over_fp32_scalar_on"]["ratio"],
    }

summary = {
    "cell": CELL,
    "issue": "PyAutoLabs/autolens_profiling#350",
    "phase": "0+1 combined (lean)",
    "config_name": config_name,
    "precision": PRECISION,
    "quotable": not config_name.startswith(("laptop", "local")),
    "host_note": (
        "laptop witness: WSL2, interactive load, tiny counts, NOT quotable"
        if config_name.startswith(("laptop", "local"))
        else None
    ),
    "quick": QUICK,
    "legs": LEGS,
    "library_versions": {
        "autolens": al.__version__,
        "autoarray": aa.__version__,
        "autofit": af.__version__,
    },
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    "source_revisions": source_revisions(_ROOT),
    "autoarray_imported_from": str(Path(aa.__file__).resolve().parent.parent),
    "autolens_imported_from": str(Path(al.__file__).resolve().parent.parent),
    "solver": {
        "construction": "al.PointSolver.for_grid(100x100 @ 0.2, precision 1e-3, mag 0.1, nd 1)",
        "n_steps": int(make_solver().n_steps),
        "max_containing_size": int(_triangles_array.MAX_CONTAINING_SIZE),
        "step0_containment": getattr(_triangles_array, "_STEP0_CONTAINMENT", None),
    },
    "device": device,
    "jax_devices": [str(d) for d in jax.devices()],
    "machine": machine_info_dict(),
    "thread_environment": thread_environment(),
    "xla_flags": os.environ.get("XLA_FLAGS"),
    "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    "jax_enable_x64": X64,
    "loadavg_start": list(_LOADAVG_START),
    "loadavg_end": list(os.getloadavg()),
    "protocol": {
        "rounds": N_ROUNDS,
        "calls": N_CALLS,
        "warm": N_WARM,
        "n_stream": N_STREAM,
        "n_eval": N_EVAL,
        "seed": SEED,
        "vmap_batches": list(VMAP_BATCHES),
        "vmap_rounds": N_VMAP_ROUNDS,
        "vmap_calls": N_VMAP_CALLS,
        "trace_calls": TRACE_CALLS,
        "trace_vmap": TRACE_VMAP,
        "steady_calls": STEADY_CALLS,
        "grad_points": N_GRAD_POINTS,
        "grad_rounds": N_GRAD_ROUNDS,
        "grad_calls": N_GRAD_CALLS,
        "grad_vmap": GRAD_VMAP,
        "vmap_delta_tol": VMAP_DELTA_TOL,
        "cache_trap_guard": "fresh closure + jax.clear_caches() per compiled program",
        "command_buffers_off": f'compiler_options={{"{_COMMAND_BUFFER_OPTION}": ""}}',
        "mdi": "90% CI half-width of a split-half bootstrap of the scalar ON per-call samples",
    },
    "headline": headline,
    "baseline": baseline,
    "stream_eval": stream_eval,
    "vmap": vmap_block,
    "trace": trace_block,
    "grad": grad_block,
    "fp32_whatif": fp32_whatif,
    "gates": gates,
    "gates_required": [g for g in required if g in gates],
    "all_gates_pass": all_gates_pass,
    "progress": progress,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_image",
    default_basename=f"{CELL}_local",
    cell=CELL,
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2, default=str))

# ---------------------------------------------------------------------------
# PNG: vmap curve, stage attribution, busy vs wall
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(ncols=3, figsize=(20, 6.5), constrained_layout=True)
ax = axes[0]
if vmap_block and vmap_block["rows"]:
    bs = [int(b) for b in vmap_block["rows"]]
    ms = [vmap_block["rows"][str(b)]["median_ms_per_likelihood"] for b in bs]
    ax.loglog(bs, ms, "o-", color="#4C72B0", label="ms / likelihood")
    ax.set_xlabel("vmap batch")
    ax.set_ylabel("ms per likelihood (median)")
    ax2 = ax.twinx()
    ax2.loglog(
        bs,
        [vmap_block["rows"][str(b)]["throughput_likelihoods_per_s"] for b in bs],
        "s--",
        color="#DD8452",
        label="likelihoods / s",
    )
    ax2.set_ylabel("throughput [likelihoods / s]")
    ax.set_title(f"vmap scaling (stopped at: {(vmap_block['stopped_at'] or {}).get('batch')})")
else:
    ax.text(0.5, 0.5, "vmap leg not run", ha="center")
ax = axes[1]
if trace_block and trace_block["programs"]:
    labels_all = []
    for prog in trace_block["programs"].values():
        for k in prog["stages"]["per_stage"]:
            if k not in labels_all:
                labels_all.append(k)
    y = np.arange(len(labels_all))
    width = 0.8 / len(trace_block["programs"])
    for j, (plabel, prog) in enumerate(trace_block["programs"].items()):
        vals = [
            prog["stages"]["per_stage"].get(k, {}).get("median_ms", 0.0) / prog["lanes"]
            for k in labels_all
        ]
        ax.barh(y + j * width, vals, height=width, label=f"{plabel} (per likelihood)")
    ax.set_yticks(y + width / 2)
    ax.set_yticklabels(labels_all, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("device kernel ms per likelihood (command buffers OFF)")
    ax.legend(fontsize=7)
    ax.set_title("stage attribution")
    ax = axes[2]
    names = list(trace_block["programs"])
    busy = [trace_block["programs"][n]["device_busy_ms_per_call"] for n in names]
    wall_off = [trace_block["programs"][n]["untraced_wall_ms"] for n in names]
    wall_on = [trace_block["programs"][n]["wall_command_buffers_on_ms"] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.25, wall_on, 0.25, label="wall, command buffers ON", color="#4C72B0")
    ax.bar(x, wall_off, 0.25, label="wall, command buffers OFF", color="#8172B3")
    ax.bar(x + 0.25, busy, 0.25, label="device busy (OFF program)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n{trace_block['programs'][n]['kernels_per_call']:.0f} kernels/call" for n in names]
    )
    ax.set_ylabel("ms per call")
    ax.legend(fontsize=7)
    ax.set_title("device busy vs wall")
else:
    ax.text(0.5, 0.5, "trace leg not run", ha="center")
    axes[2].text(0.5, 0.5, "trace leg not run", ha="center")
fig.suptitle(
    f"PointSolver bottleneck map ({config_name}); scalar ON {on_ms:.4f} ms; "
    f"all_gates_pass={all_gates_pass}",
    fontsize=10,
)
fig.savefig(chart_path, dpi=130)
plt.close(fig)

print("\n" + "=" * 110)
print(f"POINTSOLVER BOTTLENECK MAP ({config_name}, {PRECISION}, legs {LEGS})")
print("=" * 110)
print(json.dumps(_scrub(headline), indent=2, default=str))
print(f"  gates: {json.dumps({k: v.get('pass') for k, v in gates.items()})}")
print(f"  all_gates_pass: {all_gates_pass}")
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
