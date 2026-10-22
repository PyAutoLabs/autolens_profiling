"""
JAX Profiling: Cluster-Scale Robust PointSolver Settings (cluster-pointsolver phase 1)
=====================================================================================

Which ``PointSolver`` settings actually find every image at cluster scale, and
what each costs (autolens_profiling#436, Pulse campaign ``cluster-pointsolver``,
phase 1). Generalised from the single-source phase-4a instrument
``scripts/point_source_image/image_plane/solver_config_sweep.py`` to
multi-source, multi-plane solves: every source is solved on its own
``plane_redshift`` and a route returns the padded image positions of every
source, ``(n_sources, MAX_CONTAINING_SIZE, 2)``. No library code is changed:
every solver is built through ``PointSolver.for_grid`` /
``PointSolver.for_limits_and_scale``, and ``MAX_CONTAINING_SIZE`` is patched for
the duration of tracing only (the phase-4a patch, restored by value and gated).

Ground truth
------------

Every verdict rests on ``_oracle.py``: an independent lens-equation root finder
(dense residual sampling of ``traced(theta) - beta`` over a box larger than any
solver extent, Newton polish per basin, recursive local refinement near critical
curves, acceptance only at residual < 1e-10" with a certified Newton correction,
stable under a 2x finer lattice to < 1e-6", finite magnification; Jacobian
singular values recorded). It is calibrated at the start of EVERY run on the
1a analytic isothermal + shear quad and the 1c/1d ``angular_roots`` fixture and
the run aborts unless it reproduces both to < 1e-6"; the calibration is embedded
in the JSON. Never the production solver, never the simulator's stored
positions (compared, not truth), never a grouping rule.

Datasets (``--dataset``)
------------------------

``simple`` (13 components, 3 planes, 2 sources x 3 images; truth + 50 seeded
prior draws + ``--n-stress`` broad stress draws) and ``a2744`` (191 components, 8 planes, 7 sources / 24 images;
reference + 30 seeded perturbation draws). See ``_datasets.py``; the A2744
reference tracer is unavailable until one is supplied (``--a2744-reference``).

Configurations (each its own interleaved route and its own JSON row)
-------------------------------------------------------------------

- **geometry** (JIT, library ``MAX_CONTAINING_SIZE`` 20): extent {production
  grid; observed-image bounding box + the PyAutoLens#764 margin
  ``2 * scale + 3 * sigma``} x initial scale {0.5, 0.7, 1.0, 2.0}" x
  ``pixel_scale_precision`` {0.01, 0.001}. ``control`` = production extent,
  scale 1.0", precision 0.001 (the workspace cluster fits' solver).
- **diagnostic**: an over-tight extent (the bounding box shrunk to 75 % of its
  half-width, no margin) at scale 1.0" / 0.001 -- designed to miss images; it
  proves the coverage verdict can fail. Never admissible.
- **max_containing_size**: 40 and 60 at the control geometry and at scale 2.0".
  The uncapped NumPy per-step containing counts (every geometry, every draw,
  every source) are the overflow detector: overflow = an uncapped count above
  the row's cap at any step.
- **backend**: the control geometry as NumPy (dynamic shapes, uncapped), eager
  JAX (no ``jit``) and ``jax.vmap`` batches of 4 and 16. ``neighbor_degree``
  stays at its default.

Verdicts (per row; ``tol`` = the row's last-step triangle side)
---------------------------------------------------------------

For every draw and source, the oracle's accepted roots are split by
magnification against the solver's threshold 0.1 (``|mu|`` within 10 % of it is
``threshold_ambiguous`` and excluded from both sides). For each remaining root:
**covered** = exactly one distinct candidate (candidates merged at 0.005")
within ``tol``; **missed** = none; **duplicated** = more than one. A candidate
within ``tol`` of no oracle root is **spurious** (fails convergence); every
candidate's lens-equation residual is recorded. **multiplicity** = the number of
distinct candidates equals the number of considered roots. **unresolved** =
missed roots + spurious candidates. Status:

- ``admissible``: every considered root covered on every judged draw,
  multiplicity agrees everywhere, no spurious candidate, no overflow;
- ``invalid``: any of those fails (reasons listed), or the route errors;
- ``diagnostic``: a designed diagnostic row, or no reference tracer;
- ``unusable``: the row's projected wall exceeds ``--row-wall-cap``.

Two draw sets, judged separately (every row reports both): ``prior`` (the
truth / reference + seeded draws of the cell's prior) and ``stress`` (broad draws
about the reference, ``ClusterCase.stress_draws``: centres +- N(0, 0.5"), mass
parameters x (1 + N(0, 0.10)), the scaling normalisation x (1 + N(0, 0.15)),
``mass_at_200`` x 10^N(0, 0.3 dex)). Admissible needs both to pass (phase-4a
rule); a row that passes prior but fails stress is ``diagnostic`` with the
stress failure listed; a prior failure is ``invalid``.

Image count alone is never evidence. A draw whose oracle is not stable is
excluded from every row and listed. Failing rows are kept, never substituted
into a speed claim: ``best_admissible`` is a family-wise tie set
(``family_gates.sweep_tie_set``) over admissible JIT rows only.

Timing
------

Fresh closure + ``jax.clear_caches()`` per JIT route; ``compile_s``,
``first_call_s`` and the warmed per-call median with a 2000-sample paired
whole-round bootstrap 90 % CI recorded separately; routes interleaved
round-robin with the start rotated each round; ``--warm`` warm calls first.
NumPy / eager rows run ``--slow-calls`` calls per round. FLOPs, XLA
``memory_analysis`` (temp bytes) and an HLO opcode census
(``likelihood_breakdown.xla_attribution.hlo_index``) per JIT route; host, source
SHAs, dataset sha256, thread env and XLA flags in the JSON. A laptop run is
``quotable: false``: local runs are correctness smoke only.

``--pilot``: few draws and rounds; additionally prints and records the oracle
cost per draw and a per-row projected wall at the full sweep sizes
(``--full-n-draws`` / ``--full-rounds`` / ``--full-calls``), so the sweep job
can be sized.

Output: ``results/breakdown/cluster/robust_settings_sweep_<dataset>[_pilot]_<config_name>.{json,png}``.
Each row's ``per_draw[].sources[]`` keeps the verdict counts plus the positions needed
to draw them (considered roots with verdict and matched candidates, spurious,
duplicated and ignored candidates; ``_judge.per_draw_source_entry``);
``oracle_figure.py`` renders them against the oracle roots, critical curves and
caustics of each judged draw.

Memory (RAL job 412567 was OOM-killed at A2744 row 8/25, MaxRSS 66.6 GB): progress lines
after oracle calibration, each oracle draw, each uncapped-counts geometry, every row and
the timing phase carry current RSS (VmRSS), peak RSS (ru_maxrss) and the count / bytes of
``jax.live_arrays()`` (``_memory.py``). Each row stores ``rss_mb`` and ``memory``; the
summary stores ``peak_rss_mb`` and the labelled ``memory_trace``. Compiled JIT / vmap
executables are deliberately kept per row (``EXEC``) for the interleaved timing phase, so
RSS growing while live-array bytes stay flat points at retained executables (compare each
row's ``compile.memory_analysis.generated_code_size_in_bytes``).
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
_HERE = str(Path(__file__).resolve().parent)
for _p in (str(_ROOT / "scripts" / "misc"), str(_ROOT), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import collections  # noqa: E402
import contextlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from likelihood_breakdown import xla_attribution  # noqa: E402
from likelihood_breakdown.ab_verdict import AB_CONFIDENCE, MIN_AB_ROUNDS  # noqa: E402
from likelihood_breakdown.family_gates import sweep_tie_set  # noqa: E402
from likelihood_breakdown.provenance import source_revisions, thread_environment  # noqa: E402
from likelihood_breakdown.round_bootstrap import as_rounds, round_median_ratio  # noqa: E402
from likelihood_breakdown.timing import block  # noqa: E402

from _profile_cli import (  # noqa: E402
    device_info_dict,
    machine_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

_WALL_START = time.perf_counter()
_LOADAVG_START = os.getloadavg()

_cli = parse_profile_cli()
_cp = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cp.add_argument("--dataset", choices=("simple", "a2744"), default="simple")
_cp.add_argument("--pilot", action="store_true", help="few draws/rounds + wall projection")
_cp.add_argument("--n-draws", type=int, default=None, help="prior draws beyond the truth/reference")
_cp.add_argument(
    "--n-stress", type=int, default=None, help="broad stress draws (default = n-draws)"
)
_cp.add_argument("--rounds", type=int, default=None)
_cp.add_argument("--calls", type=int, default=None, help="calls per round, JIT / vmap routes")
_cp.add_argument("--slow-calls", type=int, default=1, help="calls per round, NumPy / eager")
_cp.add_argument("--warm", type=int, default=3)
_cp.add_argument("--n-stream", type=int, default=None, help="timing-stream draws")
_cp.add_argument("--configs", default=None, help="comma list of row names (control always kept)")
_cp.add_argument("--list-configs", action="store_true")
_cp.add_argument("--seed", type=int, default=436)
_cp.add_argument("--oracle-h", type=float, default=None, help="oracle lattice spacing (arcsec)")
_cp.add_argument("--oracle-half-width", type=float, default=None)
_cp.add_argument("--row-wall-cap", type=float, default=1800.0, help="seconds; above = unusable")
_cp.add_argument("--vmap-batches", default="4,16")
_cp.add_argument("--a2744-reference", default=None)
_cp.add_argument("--a2744-workspace", default=None)
_cp.add_argument("--a2744-perturbation", type=float, default=0.02)
_cp.add_argument("--full-n-draws", type=int, default=None, help="pilot projection size")
_cp.add_argument("--full-n-stress", type=int, default=None, help="pilot projection size")
_cp.add_argument("--full-rounds", type=int, default=20, help="pilot projection size")
_cp.add_argument("--full-calls", type=int, default=10, help="pilot projection size")
_args = _cli.parse_cell_args(_cp)

if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
    raise SystemExit(
        "PYAUTO_SMALL_DATASETS=1 short-circuits PointSolver.solve to a fixed pair of positions: "
        "this cell would measure nothing. Unset it; use --pilot / --n-draws / --configs instead."
    )

DATASET = _args.dataset
PILOT = bool(_args.pilot)
FULL_N_DRAWS_DEFAULT = {"simple": 50, "a2744": 30}[DATASET]
N_DRAWS = _args.n_draws if _args.n_draws is not None else (2 if PILOT else FULL_N_DRAWS_DEFAULT)
N_STRESS = _args.n_stress if _args.n_stress is not None else N_DRAWS
N_ROUNDS = _args.rounds if _args.rounds is not None else (3 if PILOT else 20)
N_CALLS = _args.calls if _args.calls is not None else (2 if PILOT else 10)
N_SLOW_CALLS = max(1, _args.slow_calls)
N_WARM = max(1, _args.warm)
N_STREAM = _args.n_stream if _args.n_stream is not None else (4 if PILOT else 8)
SEED = _args.seed
VMAP_BATCHES = tuple(int(b) for b in _args.vmap_batches.split(",") if b.strip())
ROW_WALL_CAP = float(_args.row_wall_cap)
FULL_N_DRAWS = _args.full_n_draws if _args.full_n_draws is not None else FULL_N_DRAWS_DEFAULT
FULL_N_STRESS = _args.full_n_stress if _args.full_n_stress is not None else FULL_N_DRAWS
if N_ROUNDS < 1 or N_CALLS < 1 or N_DRAWS < 0 or N_STRESS < 0 or N_STREAM < 1:
    raise SystemExit("--rounds, --calls, --n-stream must be positive; --n-draws >= 0")

matplotlib.use("Agg")
import _datasets  # noqa: E402
import _memory  # noqa: E402
import _oracle  # noqa: E402
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from _judge import (  # noqa: E402
    DISTINCT_TOL,
    MAGNIFICATION_THRESHOLD,
    MU_AMBIGUITY_BAND,
    judge,
    per_draw_source_entry,
)
from autoarray.abstract_ndarray import register_instance_pytree  # noqa: E402
from autoarray.structures.triangles import array as _triangles_array  # noqa: E402
from autoarray.structures.triangles import coordinate_array as _coordinate_array  # noqa: E402
from autoarray.structures.triangles.shape import Point  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402
from autolens.lens.tracer import Tracer  # noqa: E402

LIBRARY_MCS = int(_triangles_array.MAX_CONTAINING_SIZE)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
SCALES = (0.5, 0.7, 1.0, 2.0)
PRECISIONS = (0.01, 0.001)
MCS_ROWS = (40, 60)
TIGHT_FRACTION = 0.75

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

try:
    if DATASET == "simple":
        CASE = _datasets.load_simple(
            oracle_h=_args.oracle_h, oracle_half_width=_args.oracle_half_width
        )
    else:
        CASE = _datasets.load_a2744(
            _args.a2744_reference,
            workspace=_args.a2744_workspace,
            perturbation=_args.a2744_perturbation,
            oracle_h=_args.oracle_h,
            oracle_half_width=_args.oracle_half_width,
        )
except _datasets.ReferenceUnavailable as exc:
    print(f"REFUSING --dataset {DATASET}: {exc}")
    sys.exit(2)

POSITION_SIGMA = CASE.position_sigma
PROD = CASE.production
_PROD_GRID = al.Grid2D.uniform(
    shape_native=tuple(PROD["shape_native"]), pixel_scales=PROD["pixel_scale"]
)
PROD_LIMITS = (
    float(np.asarray(_PROD_GRID[:, 0]).min()),
    float(np.asarray(_PROD_GRID[:, 0]).max()),
    float(np.asarray(_PROD_GRID[:, 1]).min()),
    float(np.asarray(_PROD_GRID[:, 1]).max()),
)
BBOX = CASE.observed_bbox()

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def _cfg(name, block, extent, scale, precision, *, mcs=None, backend="jit", batch=None):
    return {
        "name": name,
        "block": block,
        "extent": extent,
        "scale": float(scale),
        "precision": float(precision),
        "max_containing_size": LIBRARY_MCS if mcs is None else int(mcs),
        "backend": backend,
        "vmap_batch": batch,
        "neighbor_degree": 1,
    }


def _geom_name(extent, scale, precision):
    return f"{extent}_s{scale:g}_p{precision:g}"


CONTROL_SCALE, CONTROL_PRECISION = float(PROD["pixel_scale"]), float(PROD["pixel_scale_precision"])
ALL_CONFIGS = []
for _extent in ("production", "bbox"):
    for _s in SCALES:
        for _p in PRECISIONS:
            _name = _geom_name(_extent, _s, _p)
            if _extent == "production" and _s == CONTROL_SCALE and _p == CONTROL_PRECISION:
                _name = "control"
            ALL_CONFIGS.append(_cfg(_name, "geometry", _extent, _s, _p))
ALL_CONFIGS.append(
    _cfg(
        _geom_name("tight", CONTROL_SCALE, CONTROL_PRECISION),
        "diagnostic",
        "tight",
        CONTROL_SCALE,
        CONTROL_PRECISION,
    )
)
for _s in (CONTROL_SCALE, 2.0):
    for _m in MCS_ROWS:
        _base = (
            "control" if _s == CONTROL_SCALE else _geom_name("production", _s, CONTROL_PRECISION)
        )
        ALL_CONFIGS.append(
            _cfg(
                f"{_base}_mcs{_m}",
                "max_containing_size",
                "production",
                _s,
                CONTROL_PRECISION,
                mcs=_m,
            )
        )
ALL_CONFIGS.append(
    _cfg(
        "control_numpy", "backend", "production", CONTROL_SCALE, CONTROL_PRECISION, backend="numpy"
    )
)
ALL_CONFIGS.append(
    _cfg(
        "control_eager", "backend", "production", CONTROL_SCALE, CONTROL_PRECISION, backend="eager"
    )
)
for _b in VMAP_BATCHES:
    ALL_CONFIGS.append(
        _cfg(
            f"control_vmap{_b}",
            "backend",
            "production",
            CONTROL_SCALE,
            CONTROL_PRECISION,
            backend="vmap",
            batch=_b,
        )
    )

if _args.list_configs:
    for c in ALL_CONFIGS:
        print(c)
    sys.exit(0)
CONFIGS = ALL_CONFIGS
if _args.configs:
    wanted = [w.strip() for w in _args.configs.split(",") if w.strip()]
    unknown = [w for w in wanted if w not in {c["name"] for c in ALL_CONFIGS}]
    if unknown:
        raise SystemExit(f"--configs: unknown {unknown}; see --list-configs")
    CONFIGS = [c for c in ALL_CONFIGS if c["name"] == "control" or c["name"] in wanted]
CONFIG_NAMES = [c["name"] for c in CONFIGS]
BY_NAME = {c["name"]: c for c in CONFIGS}


def limits(cfg) -> tuple:
    """(y_min, y_max, x_min, x_max) of the step-0 box a configuration tiles."""
    if cfg["extent"] == "production":
        return PROD_LIMITS
    y0, y1, x0, x1 = BBOX
    if cfg["extent"] == "bbox":
        m = 2.0 * cfg["scale"] + 3.0 * POSITION_SIGMA
        return (y0 - m, y1 + m, x0 - m, x1 + m)
    cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
    hy, hx = TIGHT_FRACTION * 0.5 * (y1 - y0), TIGHT_FRACTION * 0.5 * (x1 - x0)
    return (cy - hy, cy + hy, cx - hx, cx + hx)


def make_solver(cfg, *, use_jax: bool):
    common = dict(
        pixel_scale_precision=cfg["precision"],
        magnification_threshold=MAGNIFICATION_THRESHOLD,
        neighbor_degree=cfg["neighbor_degree"],
        use_jax=use_jax,
    )
    if cfg["extent"] == "production" and cfg["scale"] == float(PROD["pixel_scale"]):
        return al.PointSolver.for_grid(grid=_PROD_GRID, **common)  # the production construction
    y_min, y_max, x_min, x_max = limits(cfg)
    return al.PointSolver.for_limits_and_scale(
        y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max, scale=cfg["scale"], **common
    )


def geometry_key(cfg) -> tuple:
    return (cfg["extent"], cfg["scale"], cfg["precision"])


def static_description(cfg) -> dict:
    s = make_solver(cfg, use_jax=False)
    g = (float(s.y_min), float(s.y_max), float(s.x_min), float(s.x_max), float(s.scale))
    n_steps = int(s.n_steps)
    try:
        n_triangles = int(_coordinate_array._lattice_coordinates(*g).shape[0])
    except Exception as exc:  # noqa: BLE001
        n_triangles = f"unavailable: {type(exc).__name__}"
    return {
        "limits": list(g[:4]),
        "scale": g[4],
        "pixel_scale_precision": cfg["precision"],
        "n_steps": n_steps,
        "last_step_triangle_side": g[4] / 2 ** (n_steps - 1),
        "step0_triangles": n_triangles,
        "max_containing_size": cfg["max_containing_size"],
        "backend": cfg["backend"],
        "vmap_batch": cfg["vmap_batch"],
        "extent_rule": {
            "production": f'production grid {PROD["shape_native"]} @ {PROD["pixel_scale"]}"',
            "bbox": "observed-image bounding box + 2*scale + 3*sigma (PyAutoLens#764)",
            "tight": f"bounding box shrunk to {TIGHT_FRACTION:.0%} of its half-width (diagnostic)",
        }[cfg["extent"]],
    }


# ---------------------------------------------------------------------------
# MAX_CONTAINING_SIZE patch (phase-4a solver_config_sweep.py, unchanged in method)
# ---------------------------------------------------------------------------

_ORIGINAL_INIT_DEFAULTS = _triangles_array.ArrayTriangles.__init__.__defaults__
_ORIGINAL_FLS_DEFAULTS = _triangles_array.ArrayTriangles.for_limits_and_scale.__func__.__defaults__


@contextlib.contextmanager
def max_containing_size(mcs: int):
    """Set ``MAX_CONTAINING_SIZE`` for tracing: module constant + both bound defaults, restored."""
    init = _triangles_array.ArrayTriangles.__init__
    fls = _triangles_array.ArrayTriangles.for_limits_and_scale.__func__
    saved = (init.__defaults__, fls.__defaults__, _triangles_array.MAX_CONTAINING_SIZE)
    init.__defaults__ = (int(mcs),) + tuple(saved[0][1:])
    fls.__defaults__ = (int(mcs),) + tuple(saved[1][1:])
    _triangles_array.MAX_CONTAINING_SIZE = int(mcs)
    try:
        yield
    finally:
        init.__defaults__, fls.__defaults__, _triangles_array.MAX_CONTAINING_SIZE = saved


def patch_default_binding_probe(mcs: int) -> dict:
    """The patch binds BY NAME: max_containing_size reads ``mcs``, later defaults unchanged."""
    out = {"mcs": int(mcs)}
    ok = True
    for label, fn in (
        ("__init__", _triangles_array.ArrayTriangles.__init__),
        ("for_limits_and_scale", _triangles_array.ArrayTriangles.for_limits_and_scale),
    ):
        before = {
            k: v.default
            for k, v in inspect.signature(fn).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        with max_containing_size(mcs):
            during = {
                k: v.default
                for k, v in inspect.signature(fn).parameters.items()
                if v.default is not inspect.Parameter.empty
            }
        out[label] = {"during": {k: repr(v) for k, v in during.items()}}
        ok = ok and during == {**before, "max_containing_size": int(mcs)}
    with max_containing_size(mcs):
        tri = _triangles_array.ArrayTriangles(
            indices=np.zeros((1, 3), dtype=int), vertices=np.zeros((3, 2))
        )
    out["constructed_max_containing_size"] = int(tri.max_containing_size)
    out["pass"] = bool(ok and tri.max_containing_size == int(mcs))
    return out


# ---------------------------------------------------------------------------
# Small helpers (phase-4a local copies)
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
        "alias_size_in_bytes",
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


def _hlo_census(compiled):
    """Opcode counts of the optimized HLO (``xla_attribution.hlo_index``)."""
    try:
        index = xla_attribution.hlo_index(compiled.as_text())
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"
    counts = collections.Counter(i.opcode for i in index.values())
    keep = ("while", "gather", "scatter", "sort", "fusion", "custom-call", "dynamic-slice")
    return {"n_instructions": int(sum(counts.values())), **{k: int(counts.get(k, 0)) for k in keep}}


def _stats_ms(seconds) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1.0e3
    return {
        "n": int(ms.size),
        "median_ms": float(np.median(ms)),
        "p10_ms": float(np.percentile(ms, 10)),
        "p90_ms": float(np.percentile(ms, 90)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
    }


def _median_ci(seconds, n_rounds: int, seed: int) -> dict:
    """Warmed median (ms) with a paired-whole-round bootstrap 90 % interval (2000 samples)."""
    rounds = as_rounds(np.asarray(seconds, dtype=float) * 1.0e3, n_rounds)
    rng = np.random.default_rng(seed)
    boots = np.array(
        [np.median(rounds[rng.integers(0, n_rounds, n_rounds)]) for _ in range(BOOTSTRAP_SAMPLES)]
    )
    return {
        "median_ms": float(np.median(rounds)),
        "ci90_low": float(np.percentile(boots, 5)),
        "ci90_high": float(np.percentile(boots, 95)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "resampling": "paired whole rounds",
        "effective_n": int(n_rounds),
    }


_home = str(Path.home())


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str) and _home and _home != "/":
        return obj.replace(_home, "~")
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, np.generic):
        return _scrub(obj.item())
    return obj


progress = []


def note(msg):
    line = f"[{time.perf_counter() - _WALL_START:8.1f} s] {msg}"
    print(line, flush=True)
    progress.append(line)


# Memory trace (job 412567 OOM, #436): one snapshot per labelled point, printed on the
# progress line; the per-row snapshot is also stored on the row (``rss_mb``, ``memory``).
MEMORY_TRACE = []


def mem(label) -> dict:
    snap = _memory.snapshot(jax)
    MEMORY_TRACE.append({"label": label, "t_s": time.perf_counter() - _WALL_START, **snap})
    return snap


def row_mem(row, name) -> str:
    snap = mem(f"row {name}")
    row["rss_mb"] = snap["rss_mb"]
    row["memory"] = snap
    return _memory.describe(snap)


# ---------------------------------------------------------------------------
# Route functions
# ---------------------------------------------------------------------------

register_model_pytrees(CASE.model)
register_instance_pytree(Tracer, no_flatten=("cosmology",))
N_SRC = len(CASE.source_redshifts)


def solve_fn(cfg):
    """(vector, betas (n_src, 2)) -> padded positions (n_src, mcs, 2), xp = jnp."""
    solver = make_solver(cfg, use_jax=True)
    redshifts = list(CASE.source_redshifts)

    def fn(vector, betas):
        instance = CASE.model.instance_from_vector(vector=vector, xp=jnp)
        tracer = al.Tracer(galaxies=CASE.galaxies_from_instance(instance))
        out = []
        for k, z in enumerate(redshifts):
            sol = solver.solve(
                tracer=tracer,
                source_plane_coordinate=(betas[k, 0], betas[k, 1]),
                xp=jnp,
                plane_redshift=z,
                remove_infinities=False,
            )
            out.append(jnp.asarray(getattr(sol, "array", sol)))
        return jnp.stack(out)

    return fn


def numpy_solver_fn(cfg):
    """(vector, betas) -> list of (n_i, 2) finite NumPy positions per source (uncapped)."""
    solver = make_solver(cfg, use_jax=False)

    def fn(vector, betas):
        tracer = CASE.tracer_from_vector(np.asarray(vector))
        return [
            np.asarray(
                getattr(
                    s := solver.solve(
                        tracer=tracer,
                        source_plane_coordinate=(float(b[0]), float(b[1])),
                        xp=np,
                        plane_redshift=z,
                    ),
                    "array",
                    s,
                ),
                dtype=float,
            ).reshape(-1, 2)
            for b, z in zip(np.asarray(betas), CASE.source_redshifts)
        ]

    return fn


def compile_route(cfg, fn, example):
    """Fresh closure, clear caches, lower + compile under the row's MCS; first call timed."""
    jax.clear_caches()
    with max_containing_size(cfg["max_containing_size"]):
        t0 = time.perf_counter()
        lowered = jax.jit(fn).lower(*example)
        lower_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        compiled = lowered.compile()
        compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = block(compiled(*example))
    first_call_s = time.perf_counter() - t0
    shape = list(np.shape(out))
    proof = {
        "positions_shape": shape,
        "expected_rows": cfg["max_containing_size"],
        "pass": bool(
            len(shape) >= 2 and shape[-2] == cfg["max_containing_size"] and shape[-1] == 2
        ),
    }
    return compiled, {
        "lower_s": float(lower_s),
        "compile_s": float(compile_s),
        "first_call_s": float(first_call_s),
        "flops": _flops(compiled),
        "memory_analysis": _memory_analysis(compiled),
        "hlo_census": _hlo_census(compiled),
        "max_containing_size_proof": proof,
    }


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


# ``judge`` / ``per_draw_source_entry`` live in ``_judge.py`` (importable without a run).


def summarise_row(name, judged, counts_overflow) -> dict:
    """Aggregate a row's per-draw per-source verdicts."""
    flat = [j for d in judged for j in d["sources"] if j is not None]
    n = len(flat)
    tot = lambda k: int(sum(j[k] for j in flat))  # noqa: E731
    errs = [j["max_position_error"] for j in flat if j["max_position_error"] is not None]
    res = [j["max_candidate_residual"] for j in flat if j["max_candidate_residual"] is not None]
    failures = [
        {
            "draw": d["draw"],
            "source": s,
            **{
                k: j[k]
                for k in (
                    "n_roots_considered",
                    "n_candidates_distinct",
                    "n_missed",
                    "n_missed_outside_extent",
                    "n_duplicated",
                    "n_spurious",
                    "multiplicity_match",
                )
            },
            "coverage": [c for c in j["coverage"] if c["verdict"] != "covered"],
            "spurious": j["spurious"],
        }
        for d in judged
        for s, j in enumerate(d["sources"])
        if j is not None and (j["unresolved"] or j["n_duplicated"] or not j["multiplicity_match"])
    ]
    out = {
        "n_draw_sources_judged": n,
        "n_roots_considered": tot("n_roots_considered"),
        "n_covered": tot("n_covered"),
        "n_missed": tot("n_missed"),
        "n_missed_outside_extent": tot("n_missed_outside_extent"),
        "n_duplicated": tot("n_duplicated"),
        "n_spurious": tot("n_spurious"),
        "n_unresolved": tot("unresolved"),
        "n_roots_threshold_ambiguous": tot("n_roots_threshold_ambiguous"),
        "coverage_fraction": (tot("n_covered") / tot("n_roots_considered"))
        if n and tot("n_roots_considered")
        else None,
        "multiplicity_agreement": float(np.mean([j["multiplicity_match"] for j in flat]))
        if n
        else None,
        "max_position_error": max(errs) if errs else None,
        "max_candidate_residual": max(res) if res else None,
        "overflow": counts_overflow,
        "failure_examples": failures[:10],
        "n_failures": len(failures),
    }
    return out


# ===========================================================================
# Run
# ===========================================================================

note(
    f"dataset {DATASET}: {N_SRC} sources at z {CASE.source_redshifts}; reference: {CASE.reference_label}"
)

# --- 0. oracle calibration (mandatory) --------------------------------------------------
CALIBRATION = _oracle.calibrate()
note(
    f"oracle calibration pass={CALIBRATION['pass']} max error "
    f"{CALIBRATION['max_error_over_cases']:.3e} arcsec ({CALIBRATION['wall_s']:.1f} s) "
    + _memory.describe(mem("oracle calibration"))
)
if not CALIBRATION["pass"]:
    print(json.dumps(_scrub(CALIBRATION), indent=1, default=str)[:4000])
    raise SystemExit("oracle calibration FAILED: no cluster row may be judged")

# --- 1. draws, source-plane coordinates, oracle roots ------------------------------------
_prior_draws = CASE.draws(N_DRAWS, SEED)
_stress_draws, STRESS_INFO = CASE.stress_draws(N_STRESS, SEED + 2000)
DRAWS = np.concatenate([_prior_draws, _stress_draws])
SET = ["prior"] * len(_prior_draws) + ["stress"] * len(_stress_draws)
SET_INDICES = {s: [i for i, x in enumerate(SET) if x == s] for s in ("prior", "stress")}
# Index -1 for stress draws: their betas are always back-traced (never the truth centres).
BETAS = [CASE.betas(v, i if SET[i] == "prior" else -1) for i, v in enumerate(DRAWS)]
note(
    f"draw sets: prior {len(_prior_draws)} (draw 0 = reference), stress {len(_stress_draws)} "
    f"(clipped values {STRESS_INFO['n_clipped_values']})"
)
_stream_draws = CASE.draws(N_STREAM - 1, SEED + 1000)
STREAM = [(v, CASE.betas(v, i)) for i, v in enumerate(_stream_draws)]
note(f"{len(DRAWS)} judged draws (draw 0 = reference), {len(STREAM)} timing-stream draws")

ORACLE = []
oracle_cost = []
for i, (v, betas) in enumerate(zip(DRAWS, BETAS)):
    tracer = CASE.tracer_from_vector(v)
    pis = CASE.plane_indices(tracer)
    eq = _oracle.LensEquation(tracer, planes=pis)
    t0 = time.perf_counter()
    per_source = [
        _oracle.solve(eq, pi, b, CASE.oracle_box, CASE.oracle_h) for pi, b in zip(pis, betas)
    ]
    eq.clear_cache()
    oracle_cost.append(time.perf_counter() - t0)
    ORACLE.append({"draw": i, "plane_indices": pis, "sources": per_source, "equation": eq})
    note(
        f"oracle draw {i} ({SET[i]}): roots/source {[s['n_accepted'] for s in per_source]}, stable "
        f"{[s['stability']['stable'] for s in per_source]} ({oracle_cost[-1]:.1f} s) "
        + _memory.describe(mem(f"oracle draw {i}"))
    )

UNSTABLE = [
    {"draw": o["draw"], "source": k}
    for o in ORACLE
    for k, s in enumerate(o["sources"])
    if not s["stability"]["stable"]
]


def judged_source(i, k) -> bool:
    return {"draw": i, "source": k} not in UNSTABLE


# Stored dataset positions (solver output) vs the oracle's reference roots: comparison only.
stored_vs_oracle = []
for k, pos in enumerate(CASE.observed_positions):
    roots = np.asarray(
        [r["position"] for r in ORACLE[0]["sources"][k]["roots"] if r["accepted"]]
    ).reshape(-1, 2)
    d = (
        np.hypot(pos[:, None, 0] - roots[None, :, 0], pos[:, None, 1] - roots[None, :, 1]).min(
            axis=1
        )
        if roots.size
        else np.full(len(pos), np.inf)
    )
    stored_vs_oracle.append(
        {
            "source": CASE.source_names[k],
            "n_stored": int(len(pos)),
            "n_oracle": int(len(roots)),
            "stored_to_nearest_oracle_root": d.tolist(),
        }
    )

# --- 2. uncapped NumPy containing counts per geometry (overflow detector) ----------------
descriptions = {c["name"]: static_description(c) for c in CONFIGS}
geometries = {}
for c in CONFIGS:
    if c["backend"] != "numpy":
        geometries.setdefault(geometry_key(c), c)
COUNTS = {}
count_cost = {}
for key, cfg in geometries.items():
    solver = make_solver(cfg, use_jax=False)
    t0 = time.perf_counter()
    mats = []
    for i, (v, betas) in enumerate(zip(DRAWS, BETAS)):
        tracer = CASE.tracer_from_vector(v)
        per_src = []
        for b, z in zip(betas, CASE.source_redshifts):
            steps = []
            for step in solver.steps(tracer=tracer, shape=Point(*b), xp=np, plane_redshift=z):
                t = np.asarray(step.filtered_triangles.triangles)
                steps.append(int(np.isfinite(t).all(axis=(1, 2)).sum()))
            per_src.append(steps)
        mats.append(per_src)
    COUNTS[key] = mats
    count_cost[str(key)] = time.perf_counter() - t0
    note(
        f"uncapped counts {key}: max over draws/sources/steps "
        f"{max(max(s) for d in mats for s in d)} ({count_cost[str(key)]:.1f} s) "
        + _memory.describe(mem(f"uncapped counts {key}"))
    )


def overflow_for(cfg, indices=None) -> dict:
    if cfg["backend"] == "numpy":
        return {"applicable": False, "reason": "NumPy path is uncapped (dynamic shapes)"}
    cap = cfg["max_containing_size"]
    allm = COUNTS[geometry_key(cfg)]
    indices = list(range(len(allm))) if indices is None else list(indices)
    if not indices:
        return {
            "applicable": True,
            "cap": cap,
            "uncapped_max": None,
            "overflow": False,
            "n_draw_sources_overflowing": 0,
            "events": [],
            "n_draws": 0,
        }
    mats = {i: allm[i] for i in indices}
    events = [
        {
            "draw": i,
            "source": k,
            "steps_over_cap": [s for s, n in enumerate(st) if n > cap],
            "uncapped_counts_per_step": st,
        }
        for i, d in mats.items()
        for k, st in enumerate(d)
        if max(st) > cap
    ]
    n_steps = max(len(st) for d in mats.values() for st in d)
    per_step_max = [
        max((st[s] for d in mats.values() for st in d if s < len(st)), default=0)
        for s in range(n_steps)
    ]
    return {
        "applicable": True,
        "cap": cap,
        "uncapped_max_per_step": per_step_max,
        "uncapped_max": int(max(per_step_max)),
        "n_draw_sources_overflowing": len(events),
        "events": events[:10],
        "overflow": bool(events),
    }


# --- 3. every row: build, evaluate on the draws, judge -----------------------------------
ROWS = {}
EXEC = {}
for cfg in CONFIGS:
    name = cfg["name"]
    desc = descriptions[name]
    row = {
        "config": cfg,
        "static": desc,
        "status": None,
        "reasons": [],
        "reference_model": CASE.reference_label,
        "reference_available": CASE.reference_available,
    }
    ROWS[name] = row
    tol = desc["last_step_triangle_side"]
    box = desc["limits"]
    try:
        if cfg["backend"] == "jit":
            fn = solve_fn(cfg)
            ex = (jnp.asarray(DRAWS[0]), jnp.asarray(BETAS[0]))
            compiled, rec = compile_route(cfg, fn, ex)
            row["compile"] = rec
            if not rec["max_containing_size_proof"]["pass"]:
                raise RuntimeError(
                    f"MAX_CONTAINING_SIZE patch did not take: {rec['max_containing_size_proof']}"
                )
            per_call = rec["first_call_s"]

            def call(v, b, compiled=compiled):
                return np.asarray(block(compiled(jnp.asarray(v), jnp.asarray(b))))

            EXEC[name] = ("jit", compiled)
        elif cfg["backend"] == "vmap":
            B = cfg["vmap_batch"]
            fn = jax.vmap(solve_fn(cfg))
            ex = (jnp.asarray(np.stack([DRAWS[0]] * B)), jnp.asarray(np.stack([BETAS[0]] * B)))
            compiled, rec = compile_route(cfg, fn, ex)
            row["compile"] = rec
            if not rec["max_containing_size_proof"]["pass"]:
                raise RuntimeError(
                    f"vmap output shape unexpected: {rec['max_containing_size_proof']}"
                )
            per_call = rec["first_call_s"] / B
            EXEC[name] = ("vmap", compiled)
        elif cfg["backend"] == "eager":
            fn = solve_fn(cfg)
            t0 = time.perf_counter()
            with max_containing_size(cfg["max_containing_size"]):
                block(fn(jnp.asarray(DRAWS[0]), jnp.asarray(BETAS[0])))
            per_call = time.perf_counter() - t0
            row["compile"] = {
                "compile_s": None,
                "first_call_s": float(per_call),
                "note": "eager: no compile step; first call includes op-by-op dispatch caches",
            }

            def call(v, b, fn=fn, cfg=cfg):
                with max_containing_size(cfg["max_containing_size"]):
                    return np.asarray(block(fn(jnp.asarray(v), jnp.asarray(b))))

            EXEC[name] = ("eager", call)
        else:  # numpy
            fn = numpy_solver_fn(cfg)
            t0 = time.perf_counter()
            fn(DRAWS[0], BETAS[0])
            per_call = time.perf_counter() - t0
            row["compile"] = {
                "compile_s": None,
                "first_call_s": float(per_call),
                "note": "NumPy: no compile step",
            }
            call = fn
            EXEC[name] = ("numpy", fn)

        projected = per_call * (
            len(DRAWS)
            + N_WARM
            + N_ROUNDS * (N_SLOW_CALLS if cfg["backend"] in ("numpy", "eager") else N_CALLS)
        )
        row["projected_row_wall_s"] = float(projected)
        if projected > ROW_WALL_CAP:
            row["status"] = "unusable"
            row["reasons"].append(
                f"projected row wall {projected:.0f} s > --row-wall-cap {ROW_WALL_CAP:.0f} s "
                f"(first call {per_call:.2f} s)"
            )
            note(f"row {name}: UNUSABLE ({row['reasons'][-1]}) " + row_mem(row, name))
            continue

        # Evaluate on every judged draw.
        outputs = []
        if cfg["backend"] == "vmap":
            B = cfg["vmap_batch"]
            idx = list(range(len(DRAWS)))
            for start in range(0, len(idx), B):
                chunk = idx[start : start + B]
                pad = chunk + [chunk[-1]] * (B - len(chunk))
                out = np.asarray(
                    block(
                        compiled(
                            jnp.asarray(np.stack([DRAWS[j] for j in pad])),
                            jnp.asarray(np.stack([BETAS[j] for j in pad])),
                        )
                    )
                )
                outputs.extend(list(out[: len(chunk)]))
        else:
            for v, b in zip(DRAWS, BETAS):
                outputs.append(call(v, b))
        nonfinite_rows = 0
        judged = []
        for i, out in enumerate(outputs):
            srcs = []
            for k in range(N_SRC):
                cand = out[k]
                if not judged_source(i, k):
                    srcs.append(None)
                    continue
                o = ORACLE[i]
                srcs.append(
                    judge(
                        o["sources"][k],
                        cand,
                        tol,
                        box,
                        o["equation"],
                        o["plane_indices"][k],
                        BETAS[i][k],
                    )
                )
            judged.append({"draw": i, "sources": srcs})
        row["verdict"] = summarise_row(name, judged, overflow_for(cfg))
        row["verdict_by_set"] = {
            st: summarise_row(
                name,
                [d for d in judged if SET[d["draw"]] == st],
                overflow_for(cfg, SET_INDICES[st]),
            )
            for st in ("prior", "stress")
        }
        row["nonfinite_rows"] = nonfinite_rows
        row["per_draw"] = [
            {
                "draw": d["draw"],
                "set": SET[d["draw"]],
                "sources": [per_draw_source_entry(j) for j in d["sources"]],
            }
            for d in judged
        ]
    except Exception as exc:  # noqa: BLE001
        row["status"] = "invalid"
        row["reasons"].append(f"route error: {type(exc).__name__}: {exc}")
        row["traceback"] = traceback.format_exc()[-4000:]
        EXEC.pop(name, None)
        note(f"row {name}: INVALID (route error {type(exc).__name__}: {exc}) " + row_mem(row, name))
        continue

    v = row["verdict"]
    if cfg["block"] == "diagnostic":
        row["status"] = "diagnostic"
        row["reasons"].append("designed over-tight extent (diagnostic by construction)")
    elif not CASE.reference_available:
        row["status"] = "diagnostic"
        row["reasons"].append(f"no reference tracer: {CASE.reference_label}")
    else:

        def _fails(vs):
            fails = []
            if vs["n_missed"]:
                fails.append(
                    f"{vs['n_missed']} oracle roots missed "
                    f"({vs['n_missed_outside_extent']} outside extent)"
                )
            if vs["n_duplicated"]:
                fails.append(
                    f'{vs["n_duplicated"]} roots with duplicate candidates at {DISTINCT_TOL}"'
                )
            if vs["n_spurious"]:
                fails.append(
                    f"{vs['n_spurious']} spurious candidates (within tol of no oracle root)"
                )
            if vs["multiplicity_agreement"] is not None and vs["multiplicity_agreement"] < 1.0:
                fails.append(f"multiplicity agreement {vs['multiplicity_agreement']:.3f}")
            if vs["overflow"].get("overflow"):
                fails.append(
                    f"overflow: uncapped count > cap {vs['overflow']['cap']} on "
                    f"{vs['overflow']['n_draw_sources_overflowing']} draw-sources"
                )
            return fails

        vb = row["verdict_by_set"]
        prior_fails = _fails(vb["prior"])
        stress_fails = _fails(vb["stress"])
        if not vb["prior"]["n_draw_sources_judged"]:
            prior_fails.append("no judged prior draw-source (oracle unstable everywhere)")
        if prior_fails:
            # 4a rule: the prior set is the admissibility gate.
            row["status"] = "invalid"
            row["reasons"].extend(f"prior: {f}" for f in prior_fails)
            row["reasons"].extend(f"stress: {f}" for f in stress_fails)
        elif stress_fails:
            row["status"] = "diagnostic"
            row["reasons"].append("passes the prior set, fails the stress set")
            row["reasons"].extend(f"stress: {f}" for f in stress_fails)
        else:
            row["status"] = "admissible"
            row["reasons"].append(
                "prior and stress: all considered roots covered, multiplicity agrees, no overflow"
            )
    note(
        f"row {name:<26} {row['status']:<10} cov {v['n_covered']}/{v['n_roots_considered']} "
        + "".join(
            f"[{st} {vv['n_covered']}/{vv['n_roots_considered']}] "
            for st, vv in (row.get("verdict_by_set") or {}).items()
        )
        + f"miss {v['n_missed']} dup {v['n_duplicated']} spur {v['n_spurious']} "
        f"mult {v['multiplicity_agreement']} overflow {v['overflow'].get('uncapped_max')}/"
        f"{v['overflow'].get('cap')} maxerr {v['max_position_error']} " + row_mem(row, name)
    )

# --- 4. timing: interleaved rounds over every usable route -------------------------------
TIMED = [n for n in CONFIG_NAMES if n in EXEC and ROWS[n]["status"] != "unusable"]
if "control" not in TIMED:
    note("control is not timeable: no speed comparison possible")


def _route_call(name, k):
    kind, ex = EXEC[name]
    if kind == "vmap":
        B = BY_NAME[name]["vmap_batch"]
        sel = [(k * B + j) % len(STREAM) for j in range(B)]
        args = (
            jnp.asarray(np.stack([STREAM[s][0] for s in sel])),
            jnp.asarray(np.stack([STREAM[s][1] for s in sel])),
        )
        t0 = time.perf_counter()
        out = block(ex(*args))
        return (time.perf_counter() - t0) / B, out
    v, b = STREAM[k % len(STREAM)]
    if kind == "jit":
        args = (jnp.asarray(v), jnp.asarray(b))
        t0 = time.perf_counter()
        out = block(ex(*args))
        return time.perf_counter() - t0, out
    t0 = time.perf_counter()
    out = ex(v, b)
    return time.perf_counter() - t0, out


note(f"timing: {len(TIMED)} routes, warm {N_WARM}, {N_ROUNDS} rounds")
for name in TIMED:
    for w in range(N_WARM):
        _route_call(name, w)
TIMES = {n: [] for n in TIMED}
deterministic = {n: True for n in TIMED}
first_values = {}
for r in range(N_ROUNDS):
    order = TIMED[r % len(TIMED) :] + TIMED[: r % len(TIMED)] if TIMED else []
    for name in order:
        calls = N_SLOW_CALLS if BY_NAME[name]["backend"] in ("numpy", "eager") else N_CALLS
        for c in range(calls):
            k = r * calls + c
            dt, out = _route_call(name, k)
            TIMES[name].append(dt)
            if BY_NAME[name]["backend"] == "jit":
                key = k % len(STREAM)
                arr = np.asarray(out)
                if key in first_values.setdefault(name, {}):
                    if not np.array_equal(arr, first_values[name][key], equal_nan=True):
                        deterministic[name] = False
                else:
                    first_values[name][key] = arr
note("timing done " + _memory.describe(mem("timing done")))

for i, name in enumerate(TIMED):
    row = ROWS[name]
    row["per_call_ms"] = [t * 1e3 for t in TIMES[name]]
    row["stats"] = _stats_ms(TIMES[name])
    row["warmed_median"] = _median_ci(TIMES[name], N_ROUNDS, BOOTSTRAP_SEED + i)
    row["deterministic"] = deterministic[name] if BY_NAME[name]["backend"] == "jit" else None
    row["timing_unit"] = "per likelihood-equivalent multi-source solve" + (
        f" (vmap batch time / {BY_NAME[name]['vmap_batch']})"
        if BY_NAME[name]["backend"] == "vmap"
        else ""
    )
    if "control" in TIMED and name != "control":
        row["speedup_vs_control"] = round_median_ratio(
            TIMES["control"],
            TIMES[name],
            n_rounds=N_ROUNDS,
            seed=BOOTSTRAP_SEED + i,
            samples=BOOTSTRAP_SAMPLES,
        )

# --- 5. gates and best admissible ---------------------------------------------------------
gates = {
    "oracle_calibration": CALIBRATION["pass"],
    "max_containing_size_patch": {
        n: ROWS[n].get("compile", {}).get("max_containing_size_proof")
        for n in CONFIG_NAMES
        if BY_NAME[n]["backend"] in ("jit", "vmap")
    },
    "patch_restored": {
        "module_constant": int(_triangles_array.MAX_CONTAINING_SIZE),
        "pass": int(_triangles_array.MAX_CONTAINING_SIZE) == LIBRARY_MCS
        and _triangles_array.ArrayTriangles.__init__.__defaults__ == _ORIGINAL_INIT_DEFAULTS
        and _triangles_array.ArrayTriangles.for_limits_and_scale.__func__.__defaults__
        == _ORIGINAL_FLS_DEFAULTS,
    },
}
_probes = {
    m: patch_default_binding_probe(m)
    for m in sorted({c["max_containing_size"] for c in CONFIGS})
    if m != LIBRARY_MCS
}
gates["patch_keeps_later_defaults"] = {
    "probes": _probes,
    "pass": all(p["pass"] for p in _probes.values()),
}
gates["reference_truth_gate"] = CASE.provenance.get("truth_gate")
all_gates_pass = bool(
    gates["oracle_calibration"]
    and gates["patch_restored"]["pass"]
    and gates["patch_keeps_later_defaults"]["pass"]
    and all((p or {}).get("pass", True) for p in gates["max_containing_size_patch"].values())
)

_control_side = descriptions["control"]["last_step_triangle_side"]
candidates = [
    n
    for n in TIMED
    if n != "control" and ROWS[n]["status"] == "admissible" and BY_NAME[n]["backend"] == "jit"
]
precision_equivalent = {
    n: bool(descriptions[n]["last_step_triangle_side"] <= _control_side * (1 + 1e-9))
    for n in CONFIG_NAMES
}
tie = tie_any = None
if "control" in TIMED and candidates:
    timed_rows = {n: ROWS[n] for n in TIMED}
    tie = (
        sweep_tie_set(
            timed_rows,
            [n for n in candidates if precision_equivalent[n]],
            n_rounds=N_ROUNDS,
            seed=BOOTSTRAP_SEED,
            samples=BOOTSTRAP_SAMPLES,
            min_n=MIN_AB_ROUNDS,
            confidence=AB_CONFIDENCE,
        )
        if any(precision_equivalent[n] for n in candidates)
        else None
    )
    tie_any = sweep_tie_set(
        timed_rows,
        candidates,
        n_rounds=N_ROUNDS,
        seed=BOOTSTRAP_SEED,
        samples=BOOTSTRAP_SAMPLES,
        min_n=MIN_AB_ROUNDS,
        confidence=AB_CONFIDENCE,
    )

# --- 6. pilot projection --------------------------------------------------------------------
pilot = None
if PILOT:
    per_draw_oracle = float(np.mean(oracle_cost)) if oracle_cost else None
    per_draw_counts = {k: v / len(DRAWS) for k, v in count_cost.items()}
    rows_proj = {}
    for name in CONFIG_NAMES:
        r = ROWS[name]
        comp = r.get("compile") or {}
        first = comp.get("first_call_s") or 0.0
        cstep = comp.get("compile_s") or 0.0
        per_call = (r.get("stats") or {}).get("median_ms")
        per_call = per_call / 1e3 if per_call is not None else first
        B = BY_NAME[name]["vmap_batch"] or 1
        calls = N_SLOW_CALLS if BY_NAME[name]["backend"] in ("numpy", "eager") else _args.full_calls
        rows_proj[name] = {
            "compile_s": cstep,
            "first_call_s": first,
            "per_call_s": per_call,
            "projected_s": cstep
            + first
            + per_call * (FULL_N_DRAWS + 1 + FULL_N_STRESS)
            + per_call * (_args.full_rounds * calls + N_WARM),
            "status_in_pilot": r["status"],
        }
    total = (
        CALIBRATION["wall_s"]
        + (per_draw_oracle or 0.0) * (FULL_N_DRAWS + 1 + FULL_N_STRESS)
        + sum(v * (FULL_N_DRAWS + 1 + FULL_N_STRESS) for v in per_draw_counts.values())
        + sum(v["projected_s"] for v in rows_proj.values())
    )
    pilot = {
        "note": "projection from THIS host; a laptop pilot is a code-path check, never a wall basis",
        "full_sizes": {
            "n_draws": FULL_N_DRAWS,
            "n_stress": FULL_N_STRESS,
            "rounds": _args.full_rounds,
            "calls": _args.full_calls,
        },
        "oracle_s_per_draw": per_draw_oracle,
        "uncapped_counts_s_per_draw": per_draw_counts,
        "rows": rows_proj,
        "projected_total_s": total,
    }
    print("\nPILOT projection (this host):")
    print(
        f"  oracle per draw (all sources): {per_draw_oracle:.1f} s; calibration {CALIBRATION['wall_s']:.1f} s"
    )
    for k, v in per_draw_counts.items():
        print(f"  uncapped counts per draw {k}: {v:.2f} s")
    for n, v in rows_proj.items():
        print(
            f"  {n:<28} compile {v['compile_s']:8.2f} s first {v['first_call_s']:8.3f} s "
            f"per-call {v['per_call_s']:9.4f} s -> {v['projected_s']:8.1f} s [{v['status_in_pilot']}]"
        )
    print(
        f"  PROJECTED full sweep: {total:.0f} s ({total / 3600:.2f} h) at {FULL_N_DRAWS} prior + {FULL_N_STRESS} stress draws, "
        f"{_args.full_rounds} rounds x {_args.full_calls} calls"
    )

# --- 7. summary JSON + PNG ------------------------------------------------------------------
CELL = f"robust_settings_sweep_{DATASET}" + ("_pilot" if PILOT else "")
config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_fp64"
)
device = device_info_dict()
if device.get("jax_compilation_cache_dir"):
    device["jax_compilation_cache_dir"] = (
        "<scrubbed>/" + Path(device["jax_compilation_cache_dir"]).name
    )

table = []
for name in CONFIG_NAMES:
    r = ROWS[name]
    v = r.get("verdict") or {}
    wm = r.get("warmed_median") or {}
    comp = r.get("compile") or {}
    ma = comp.get("memory_analysis")
    table.append(
        {
            "config": name,
            "block": BY_NAME[name]["block"],
            "backend": BY_NAME[name]["backend"],
            "extent": BY_NAME[name]["extent"],
            "scale": BY_NAME[name]["scale"],
            "precision": BY_NAME[name]["precision"],
            "mcs": BY_NAME[name]["max_containing_size"],
            "n_steps": r["static"]["n_steps"],
            "tol_last_step_side": r["static"]["last_step_triangle_side"],
            "status": r["status"],
            "reasons": r["reasons"],
            "reference_model": CASE.reference_label,
            "coverage": f"{v.get('n_covered')}/{v.get('n_roots_considered')}" if v else None,
            "by_set": {
                st: {
                    "coverage": f"{vv['n_covered']}/{vv['n_roots_considered']}",
                    "n_missed": vv["n_missed"],
                    "n_duplicated": vv["n_duplicated"],
                    "n_spurious": vv["n_spurious"],
                    "n_unresolved": vv["n_unresolved"],
                    "multiplicity_agreement": vv["multiplicity_agreement"],
                    "overflow": vv["overflow"].get("overflow"),
                    "uncapped_max": vv["overflow"].get("uncapped_max"),
                }
                for st, vv in (r.get("verdict_by_set") or {}).items()
            },
            "n_missed": v.get("n_missed"),
            "n_duplicated": v.get("n_duplicated"),
            "n_spurious": v.get("n_spurious"),
            "n_unresolved": v.get("n_unresolved"),
            "multiplicity_agreement": v.get("multiplicity_agreement"),
            "overflow": (v.get("overflow") or {}).get("overflow"),
            "uncapped_max": (v.get("overflow") or {}).get("uncapped_max"),
            "max_position_error": v.get("max_position_error"),
            "max_candidate_residual": v.get("max_candidate_residual"),
            "compile_s": comp.get("compile_s"),
            "first_call_s": comp.get("first_call_s"),
            "warmed_median_ms": wm.get("median_ms"),
            "warmed_ci90_ms": [wm.get("ci90_low"), wm.get("ci90_high")] if wm else None,
            "speedup_vs_control": (r.get("speedup_vs_control") or {}).get("ratio"),
            "flops": comp.get("flops"),
            "temp_bytes": ma.get("temp_size_in_bytes") if isinstance(ma, dict) else None,
            "precision_equivalent": precision_equivalent[name],
        }
    )

dataset_digests = CASE.provenance.get("files_sha256")
summary = {
    "cell": CELL,
    "issue": "PyAutoLabs/autolens_profiling#436",
    "campaign": "cluster-pointsolver phase 1",
    "dataset": DATASET,
    "pilot": PILOT,
    "config_name": config_name,
    "quotable": bool(not config_name.startswith(("laptop", "local")) and CASE.reference_available),
    "host_note": "local run: correctness smoke only, timings NOT a measurement"
    if config_name.startswith(("laptop", "local"))
    else None,
    "reference": {
        "label": CASE.reference_label,
        "available": CASE.reference_available,
        "vector": CASE.reference_vector.tolist(),
        "vector_paths": [".".join(p) for p in CASE.model.unique_prior_paths],
    },
    "dataset_provenance": CASE.provenance,
    "dataset_sha256": dataset_digests,
    "sources": [
        {"name": n, "redshift": z, "observed_positions": p.tolist()}
        for n, z, p in zip(CASE.source_names, CASE.source_redshifts, CASE.observed_positions)
    ],
    "position_sigma": POSITION_SIGMA,
    "production_solver": PROD,
    "observed_bbox": list(BBOX),
    "oracle": {
        "calibration": CALIBRATION,
        "parameters": _oracle.PARAMETERS,
        "box": list(CASE.oracle_box),
        "h": CASE.oracle_h,
        "s_per_draw": oracle_cost,
        "unstable_draw_sources": UNSTABLE,
        "per_draw": [
            {
                "draw": o["draw"],
                "plane_indices": o["plane_indices"],
                "betas": BETAS[o["draw"]].tolist(),
                "sources": [
                    {k: s[k] for k in s if k != "search"} | {"search": s.get("search")}
                    for s in o["sources"]
                ],
            }
            for o in ORACLE
        ],
        "stored_positions_vs_reference_roots": stored_vs_oracle,
    },
    "draws": {
        "n_judged": len(DRAWS),
        "n_prior": len(SET_INDICES["prior"]),
        "n_stress": len(SET_INDICES["stress"]),
        "set_per_draw": SET,
        "stress": STRESS_INFO,
        "kind": CASE.draw_kind,
        "seed": SEED,
        "perturbation_fraction": CASE.perturbation_fraction,
        "vectors": DRAWS.tolist(),
        "n_stream": len(STREAM),
        "stream_seed": SEED + 1000,
    },
    "uncapped_counts": {str(k): v for k, v in COUNTS.items()},
    "uncapped_counts_s": count_cost,
    "library_versions_note": "al.__version__ on a source checkout is stale; source_revisions are provenance",
    "source_revisions": source_revisions(_ROOT),
    "autoarray_imported_from": str(Path(aa.__file__).resolve().parent.parent),
    "autolens_imported_from": str(Path(al.__file__).resolve().parent.parent),
    "autofit_imported_from": str(Path(af.__file__).resolve().parent.parent),
    "package_versions": {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
    },
    "device": device,
    "jax_devices": [str(d) for d in jax.devices()],
    "machine": machine_info_dict(),
    "thread_environment": thread_environment(),
    "sched_affinity": len(os.sched_getaffinity(0)),
    "xla_flags": os.environ.get("XLA_FLAGS"),
    "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "loadavg_start": list(_LOADAVG_START),
    "loadavg_end": list(os.getloadavg()),
    "protocol": {
        "n_prior_draws": N_DRAWS,
        "n_stress_draws": N_STRESS,
        "admissibility": "prior AND stress pass (4a rule); prior pass + stress fail = diagnostic",
        "n_rounds": N_ROUNDS,
        "n_calls": N_CALLS,
        "n_slow_calls": N_SLOW_CALLS,
        "n_warm": N_WARM,
        "row_wall_cap_s": ROW_WALL_CAP,
        "vmap_batches": list(VMAP_BATCHES),
        "route_order": "round-robin over every usable route, start rotated each round",
        "cache_trap_guard": "fresh closure + jax.clear_caches() per JIT route",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "90%",
            "resampling": "paired whole rounds",
        },
        "verdict_rules": {
            "tolerance": "row's last-step triangle side = scale / 2**(n_steps - 1)",
            "distinct_tol": DISTINCT_TOL,
            "magnification_threshold": MAGNIFICATION_THRESHOLD,
            "threshold_ambiguity_band": MU_AMBIGUITY_BAND,
            "admissible": "every considered oracle root covered by exactly one distinct candidate within "
            "tol on every judged draw-source, multiplicity agrees, no spurious candidate, "
            "no uncapped count above the row's cap",
        },
        "max_containing_size_patch": "ArrayTriangles.__init__ / for_limits_and_scale __defaults__ + module "
        "constant, tracing (and eager calls) only, restored by value",
    },
    "configs": CONFIGS,
    "table": table,
    "rows": ROWS,
    "best_admissible": tie.best if tie else None,
    "best_admissible_tie_set": tie.as_dict() if tie else None,
    "best_admissible_any_precision_tie_set": tie_any.as_dict() if tie_any else None,
    "precision_equivalent": precision_equivalent,
    "gates": gates,
    "all_gates_pass": all_gates_pass,
    "pilot_projection": pilot,
    "progress": progress,
    "peak_rss_mb": _memory.snapshot()["peak_rss_mb"],
    "memory_trace": MEMORY_TRACE,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "cluster",
    default_basename=f"{CELL}_local",
    cell=CELL,
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2, default=str))

_COLOURS = {
    "admissible": "#4C72B0",
    "invalid": "#C44E52",
    "diagnostic": "#8C8C8C",
    "unusable": "#DD8452",
}
fig, (ax1, ax2) = plt.subplots(
    ncols=2, figsize=(16, max(6, 0.32 * len(table) + 2)), constrained_layout=True
)
ys = np.arange(len(table))
for y, t in zip(ys, table):
    c = _COLOURS.get(t["status"], "black")
    if t["warmed_median_ms"] is not None:
        ax1.barh(y, t["warmed_median_ms"], color=c, alpha=0.8)
        if t["warmed_ci90_ms"] and None not in t["warmed_ci90_ms"]:
            ax1.plot(t["warmed_ci90_ms"], [y, y], color="black", lw=1)
    frac = None
    if t["coverage"] and "/" in t["coverage"] and t["coverage"].split("/")[1] not in ("0", "None"):
        a, b = t["coverage"].split("/")
        frac = int(a) / int(b) if a != "None" else None
    ax2.barh(y, frac if frac is not None else 0.0, color=c, alpha=0.8)
for ax in (ax1, ax2):
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t['config']} [{t['status']}]" for t in table], fontsize=7)
    ax.invert_yaxis()
ax1.set_xscale("log")
ax1.set_xlabel("warmed median per multi-source solve [ms] (90% CI); colour = status")
ax2.set_xlabel("oracle-root coverage fraction over judged draws (colour = status)")
ax2.set_xlim(0, 1.05)
fig.suptitle(
    f"Cluster PointSolver robust settings ({DATASET}, {config_name}{', PILOT' if PILOT else ''}); "
    f'oracle calibration {CALIBRATION["pass"]} (max err {CALIBRATION["max_error_over_cases"]:.1e}"); '
    f"best admissible={summary['best_admissible']}; gates={all_gates_pass}",
    fontsize=9,
)
fig.savefig(chart_path, dpi=130)
plt.close(fig)

print("\n" + "=" * 140)
print(
    f"CLUSTER ROBUST SETTINGS ({DATASET}, {config_name}{', PILOT' if PILOT else ''}): "
    f"{len(DRAWS)} draws, {N_ROUNDS} rounds"
)
print("=" * 140)
for t in table:
    print(
        f"  {t['config']:<28} {t['status']:<10} cov {str(t['coverage']):>7} miss {t['n_missed']} "
        f"dup {t['n_duplicated']} spur {t['n_spurious']} mult {t['multiplicity_agreement']} "
        f"ovf {t['overflow']} ({t['uncapped_max']}/{t['mcs']}) tol {t['tol_last_step_side']:.2e} "
        + " ".join(
            f"{st}:{b['coverage']} m{b['multiplicity_agreement']} o{b['uncapped_max']}"
            for st, b in t["by_set"].items()
        )
    )
print(f"  unstable oracle draw-sources: {UNSTABLE}")
print(f"  best admissible: {summary['best_admissible']}; all_gates_pass: {all_gates_pass}")
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
