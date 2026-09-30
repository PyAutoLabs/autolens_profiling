"""
Stage map for the JAX ``PointSolver`` image-plane likelihood (point-source GPU phase 0+1).

The imaging stage map in ``scripts/misc/likelihood_breakdown/xla_attribution.py`` names
inversion stages; this module is its point-solver sibling. It builds
:class:`~likelihood_breakdown.xla_attribution.StageRule` rules for the production
``FitPositionsImagePairAllSolved`` likelihood (``AnalysisPoint`` + ``PointSolver.for_grid``)
and adds the one thing the imaging map never needed: a **lattice / active split** of the
solver-step stages by array size.

Why a size split, not a per-step split
--------------------------------------

``AbstractSolver.steps`` is a Python ``for number in range(self.n_steps)`` loop, so under
``jit`` it is *unrolled*: every refinement step traces the same source lines of
``shape_solver.py`` / ``coordinate_array.py`` / ``array.py``. The HLO source stack therefore
cannot tell step 3 from step 5 -- nothing in the metadata carries the loop index. What the
arrays DO carry is their size. Step 0 works on the static lattice (for the production
+-9.9" / 0.2" grid: 23 283 triangles, 11 859 unique vertices); every later step -- and the
neighbourhood / up-sample of step 0's kept set -- works on the ``MAX_CONTAINING_SIZE``-padded
active set (at MCS 20: 20 kept -> 80 neighbourhood -> 320 up-sampled triangles, 960 flat
vertices). An instruction whose output or any fused constituent has a dimension of at least
``lattice_min_dim`` is ``*_lattice``; otherwise ``*_active``. The step-0 containment route
has its own function (``ArrayTriangles._step0_point_mask``) and is labelled from source.

Rule order and contexts
-----------------------

``xla_attribution.stage_for_frames`` walks frames innermost-first and tries the rules in
order at each frame. A deflection is innermost in an ``autogalaxy`` mass profile whichever
caller asked for it, so the caller is recovered with ``requires`` contexts, resolved from
the *installed* library by :func:`library_ranges` (``inspect.getsourcelines``) rather than
pinned line numbers:

1. solver-step deflections (``AbstractSolver._plane_grid`` in the stack),
2. the magnification filter (``AbstractSolver._filter_low_magnification``),
3. the solved source centre beta* (``autolens/point/fit/solved.py``),
4. the implicit-gradient JVP (``autolens/point/solver/implicit_diff.py``) -- only reached
   when none of the primal contexts above is in the stack, so the primal solve *inside*
   the ``custom_jvp`` is still attributed to its solver stages,
5. the chi-squared pairing (``autolens/point/fit/positions/``).

Everything else falls to ``other`` and is listed by source in the attribution's census, so
a run says what the map missed.

This module imports neither JAX nor the PyAuto* libraries at import time; only
:func:`library_ranges` imports them, so the rules are unit-testable on hand-built frames.
"""

from __future__ import annotations

import inspect
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_MISC = str(_profiling_root() / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

from likelihood_breakdown import xla_attribution as xa  # noqa: E402

_TRI = "autoarray/structures/triangles/"
_ARRAY = _TRI + "array.py"
_COORD = _TRI + "coordinate_array.py"
_ABSTRACT_TRI = _TRI + "abstract.py"
_SHAPE = _TRI + "shape.py"
_SOLVER = "autolens/point/solver/"
_SHAPE_SOLVER = _SOLVER + "shape_solver.py"
_POINT_SOLVER = _SOLVER + "point_solver.py"
_IMPLICIT = _SOLVER + "implicit_diff.py"
_FIT = "autolens/point/fit/"
_SOLVED = _FIT + "solved.py"
_POSITIONS = _FIT + "positions/"
_ANALYSIS = "autolens/point/model/analysis.py"

#: Library code that computes a deflection / Hessian / ray trace for whichever caller.
_PHYSICS_PATHS = ("autogalaxy/", "autoarray/structures/grids/", "autolens/lens/")

#: Production step-0 lattice: 23 283 triangles, 11 859 unique vertices; the active set at
#: MCS 20 never exceeds 4 * 4 * 20 * 3 = 960 rows. 4000 separates them with margin, and a
#: vmap batch axis (<= 256) cannot cross it.
LATTICE_MIN_DIM = 4000

#: Labels that are split into ``<label>_lattice`` / ``<label>_active`` by array size.
SIZE_SPLIT_LABELS = frozenset(
    {
        "containment",
        "deflections",
        "kept_gather",
        "neighbourhood",
        "up_sample",
        "triangle_geometry",
    }
)

#: Display order for tables and the PNG.
STAGE_ORDER = (
    "step0_containment",
    "containment_lattice",
    "containment_active",
    "deflections_lattice",
    "deflections_active",
    "kept_gather_lattice",
    "kept_gather_active",
    "neighbourhood_lattice",
    "neighbourhood_active",
    "up_sample_lattice",
    "up_sample_active",
    "triangle_geometry_lattice",
    "triangle_geometry_active",
    "magnification_filter",
    "solver_output",
    "beta_star",
    "chi_squared",
    "implicit_jvp",
    "model_mapping",
    "analysis_other",
    xa.MIXED_FUSION,
    xa.OTHER,
    xa.MEMSET,
    xa.UNJOINED,
)

#: Every library function a rule names, as ``key -> (module, qualname)``. Rules match on
#: the function's LINE RANGE in the installed source (resolved by :func:`library_ranges`),
#: so they hold whether the stack frame index records ``co_name`` or ``co_qualname``.
LIBRARY_FUNCTIONS = {
    "plane_grid": ("autolens.point.solver.shape_solver", "AbstractSolver._plane_grid"),
    "plane_triangles": ("autolens.point.solver.shape_solver", "AbstractSolver._plane_triangles"),
    "filter_low_magnification": (
        "autolens.point.solver.shape_solver",
        "AbstractSolver._filter_low_magnification",
    ),
    "step0_point_mask": (
        "autoarray.structures.triangles.array",
        "ArrayTriangles._step0_point_mask",
    ),
    "containing_indices": (
        "autoarray.structures.triangles.array",
        "ArrayTriangles.containing_indices",
    ),
    "array_with_vertices": ("autoarray.structures.triangles.array", "ArrayTriangles.with_vertices"),
    "array_for_indexes": ("autoarray.structures.triangles.array", "ArrayTriangles.for_indexes"),
    "array_neighborhood": ("autoarray.structures.triangles.array", "ArrayTriangles.neighborhood"),
    "array_neighborhood_triangles": (
        "autoarray.structures.triangles.array",
        "ArrayTriangles._neighborhood_triangles",
    ),
    "array_up_sample": ("autoarray.structures.triangles.array", "ArrayTriangles.up_sample"),
    "array_up_sample_triangle": (
        "autoarray.structures.triangles.array",
        "ArrayTriangles._up_sample_triangle",
    ),
    "remove_duplicates": ("autoarray.structures.triangles.array", "remove_duplicates"),
    "select_and_handle_invalid": (
        "autoarray.structures.triangles.array",
        "select_and_handle_invalid",
    ),
    "coord_with_vertices": (
        "autoarray.structures.triangles.coordinate_array",
        "CoordinateArrayTriangles.with_vertices",
    ),
    "coord_for_indexes": (
        "autoarray.structures.triangles.coordinate_array",
        "CoordinateArrayTriangles.for_indexes",
    ),
    "coord_neighborhood": (
        "autoarray.structures.triangles.coordinate_array",
        "CoordinateArrayTriangles.neighborhood",
    ),
    "coord_up_sample": (
        "autoarray.structures.triangles.coordinate_array",
        "CoordinateArrayTriangles.up_sample",
    ),
}


def library_ranges() -> dict[str, tuple[int, int]]:
    """``(first, last)`` source lines of every :data:`LIBRARY_FUNCTIONS` entry, INSTALLED.

    Resolved with ``inspect`` so the stage map follows the code that ran rather than a
    pinned revision (the imaging map's pinned ranges needed a separate anchor check).
    """
    import importlib

    out: dict[str, tuple[int, int]] = {}
    for key, (module_name, qualname) in LIBRARY_FUNCTIONS.items():
        obj = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(getattr(obj, "fget", obj))
        lines, start = inspect.getsourcelines(obj)
        out[key] = (int(start), int(start + len(lines) - 1))
    return out


def build_stage_map(ranges: Mapping[str, tuple[int, int]]) -> tuple[xa.StageRule, ...]:
    """Ordered :class:`StageRule` tuple for the point-solver likelihood.

    *ranges* maps every :data:`LIBRARY_FUNCTIONS` key to a ``(first, last)`` line range
    (:func:`library_ranges` for the installed library; tests pass their own).
    """

    def r(key):
        return tuple(ranges[key])

    rules: list[xa.StageRule] = [
        # --- step-0 containment route (source-identified) ---------------------------------
        xa.StageRule("step0_containment", _ARRAY, lines=r("step0_point_mask")),
        xa.StageRule("step0_containment", _SHAPE, requires=(_ARRAY, r("step0_point_mask"))),
        # --- containment on every other triangle set ---------------------------------------
        xa.StageRule("containment", _SHAPE),
        xa.StageRule("containment", _ARRAY, lines=r("containing_indices")),
    ]
    # --- caller contexts for physics code (deflections, Hessians, ray traces) -------------
    for label, context in (
        ("deflections", (_SHAPE_SOLVER, r("plane_grid"))),
        ("magnification_filter", (_SHAPE_SOLVER, r("filter_low_magnification"))),
        ("beta_star", (_SOLVED, None)),
        ("implicit_jvp", (_IMPLICIT, None)),
        ("chi_squared", (_POSITIONS, None)),
    ):
        for path in _PHYSICS_PATHS:
            rules.append(xa.StageRule(label, path, requires=context))
    by_function = (
        ("deflections", _SHAPE_SOLVER, "plane_grid"),
        ("deflections", _SHAPE_SOLVER, "plane_triangles"),
        ("deflections", _ARRAY, "array_with_vertices"),
        ("deflections", _COORD, "coord_with_vertices"),
        ("magnification_filter", _SHAPE_SOLVER, "filter_low_magnification"),
        ("kept_gather", _COORD, "coord_for_indexes"),
        ("kept_gather", _ARRAY, "array_for_indexes"),
        ("neighbourhood", _COORD, "coord_neighborhood"),
        ("neighbourhood", _ARRAY, "array_neighborhood"),
        ("neighbourhood", _ARRAY, "array_neighborhood_triangles"),
        ("up_sample", _COORD, "coord_up_sample"),
        ("up_sample", _ARRAY, "array_up_sample"),
        ("up_sample", _ARRAY, "array_up_sample_triangle"),
        ("up_sample", _ARRAY, "remove_duplicates"),
        ("up_sample", _ARRAY, "select_and_handle_invalid"),
    )
    rules += [xa.StageRule(label, path, lines=r(key)) for label, path, key in by_function]
    rules += [
        # --- whole-file fallbacks, most specific first ---------------------------------------
        xa.StageRule("triangle_geometry", _COORD),
        xa.StageRule("triangle_geometry", _ARRAY),
        xa.StageRule("triangle_geometry", _ABSTRACT_TRI),
        xa.StageRule("solver_output", _POINT_SOLVER),
        xa.StageRule("solver_output", _SHAPE_SOLVER),
        xa.StageRule("implicit_jvp", _IMPLICIT),
        xa.StageRule("beta_star", _SOLVED),
        xa.StageRule("chi_squared", _POSITIONS),
        xa.StageRule("analysis_other", _ANALYSIS),
        xa.StageRule("analysis_other", _FIT),
        xa.StageRule("model_mapping", "autofit/"),
    ]
    return tuple(rules)


_ARRAY_SHAPE_RE = re.compile(r"\b[a-z]+\d*\[(?P<dims>[\d,]*)\]")


def shape_dims(shape: str) -> list[int]:
    """Every dimension of every array in an HLO shape string (tuples included).

    Local rather than ``Instruction.dims``, whose dtype pattern ``[a-z]\\d*`` does not match
    ``pred[...]`` (every containment mask) or a tuple-shaped output.
    """
    out: list[int] = []
    for match in _ARRAY_SHAPE_RE.finditer(shape or ""):
        out.extend(int(d) for d in match.group("dims").split(",") if d.strip())
    return out


def max_dim(instruction: xa.Instruction, index: Mapping[str, xa.Instruction]) -> int:
    """Largest single dimension over the instruction's output and every fused constituent."""
    dims = shape_dims(instruction.shape)
    for member in xa.constituents_of(instruction, index):
        dims.extend(shape_dims(member.shape))
    return max(dims) if dims else 0


def classify(
    instruction: xa.Instruction,
    index: Mapping[str, xa.Instruction],
    stage_map: Sequence[xa.StageRule],
    lattice_min_dim: int = LATTICE_MIN_DIM,
) -> str:
    """Stage label of one instruction, with the lattice / active split applied."""
    label, _members = xa.stages_of_instruction(instruction, index, stage_map)
    if label in SIZE_SPLIT_LABELS:
        suffix = "lattice" if max_dim(instruction, index) >= lattice_min_dim else "active"
        return f"{label}_{suffix}"
    return label


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def inter_kernel_gaps_ms(events: Sequence[dict]) -> list[float]:
    """Gaps between the union of kernel intervals, in time order (ms).

    Local rather than ``xla_attribution.idle_gaps``, whose sort key falls through to
    comparing event dicts when two kernels share a start and an end (seen on the CPU
    witness, whose thunks run on several threads).
    """
    intervals = sorted((e["start_ns"], e["start_ns"] + e["dur_ns"]) for e in events)
    gaps: list[float] = []
    cur_end = None
    for start, end in intervals:
        if cur_end is not None and start > cur_end:
            gaps.append((start - cur_end) / 1e6)
        if cur_end is None or end > cur_end:
            cur_end = end
    return gaps


def stage_table(
    events: Sequence[dict],
    index: Mapping[str, xa.Instruction],
    *,
    calls: int,
    stage_map: Sequence[xa.StageRule],
    lattice_min_dim: int = LATTICE_MIN_DIM,
) -> dict:
    """Per-call kernel ms AND kernel count per stage (lattice / active split applied).

    Every device event lands in exactly one row: an event with no ``hlo_op`` is
    ``memset`` (or ``unjoined`` if it is not a memset), an ``hlo_op`` that is not an
    instruction of *index* is ``unjoined``. Per-call values are medians over the calls
    when :func:`xla_attribution.split_calls` can split the trace, else the trace total
    divided by *calls* (``call_split`` says which).
    """
    cache: dict[str, str] = {}

    def label_of(event: dict) -> str:
        op = event.get("hlo_op")
        if op is None:
            return xa.MEMSET if "memset" in event["name"].lower() else xa.UNJOINED
        if op not in index:
            return xa.UNJOINED
        if op not in cache:
            cache[op] = classify(index[op], index, stage_map, lattice_min_dim)
        return cache[op]

    blocks, method = xa.split_calls(events, calls)
    divisor = calls if len(blocks) == 1 else 1
    per_call_ms: list[dict[str, float]] = []
    per_call_n: list[dict[str, int]] = []
    for block in blocks:
        ms: dict[str, float] = {}
        n: dict[str, int] = {}
        for event in block:
            label = label_of(event)
            ms[label] = ms.get(label, 0.0) + event["dur_ns"] / 1e6
            n[label] = n.get(label, 0) + 1
        per_call_ms.append(ms)
        per_call_n.append(n)

    labels = {label for row in per_call_ms for label in row}
    rows = {}
    for label in labels:
        ms_series = [row.get(label, 0.0) / divisor for row in per_call_ms]
        n_series = [row.get(label, 0) / divisor for row in per_call_n]
        rows[label] = {
            "median_ms": _median(ms_series),
            "min_ms": min(ms_series),
            "max_ms": max(ms_series),
            "kernels": _median(n_series),
        }
    kernel_ms = sum(r["median_ms"] for r in rows.values())
    for r in rows.values():
        r["share_of_kernel_pct"] = 100.0 * r["median_ms"] / kernel_ms if kernel_ms else 0.0

    # Launch-overhead fingerprint: the gaps between consecutive kernels inside one call.
    gaps_us: list[float] = []
    busy_ms: list[float] = []
    span_ms: list[float] = []
    for block in blocks if divisor == 1 else []:
        gaps_us.extend(g * 1e3 for g in inter_kernel_gaps_ms(block))
        busy_ms.append(xa._union_busy_ns(block) / 1e6)
        span_ms.append(
            (max(e["start_ns"] + e["dur_ns"] for e in block) - min(e["start_ns"] for e in block))
            / 1e6
        )
    kernel_counts = [sum(n.values()) / divisor for n in per_call_n]
    return {
        "call_split": method,
        "call_blocks": len(blocks),
        "kernels_per_call": _median(kernel_counts),
        "kernels_per_call_min": min(kernel_counts) if kernel_counts else 0,
        "kernels_per_call_max": max(kernel_counts) if kernel_counts else 0,
        "kernel_ms_per_call": kernel_ms,
        "per_stage": {
            label: rows[label]
            for label in [s for s in STAGE_ORDER if s in rows]
            + sorted(set(rows) - set(STAGE_ORDER))
        },
        "device_busy_ms_per_call_median": _median(busy_ms) if busy_ms else None,
        "device_span_ms_per_call_median": _median(span_ms) if span_ms else None,
        "inter_kernel_gap_us": {
            "n": len(gaps_us),
            "median": _median(gaps_us),
            "p10": _percentile(gaps_us, 10),
            "p90": _percentile(gaps_us, 90),
            "max": max(gaps_us) if gaps_us else 0.0,
            "sum_per_call_ms": (sum(gaps_us) / 1e3 / len(blocks))
            if blocks and divisor == 1
            else None,
        },
        "lattice_min_dim": lattice_min_dim,
        "note": (
            "Kernel ms and kernel counts per call, per stage, on the command-buffers-OFF "
            "program. '_lattice' = an instruction touching an array with a dimension >= "
            "lattice_min_dim (the step-0 static lattice); '_active' = the MCS-padded active set "
            "of every refinement step (and step 0's neighbourhood / up-sample). The unrolled "
            "Python step loop shares source lines, so no per-step split exists in the HLO "
            "metadata."
        ),
    }
