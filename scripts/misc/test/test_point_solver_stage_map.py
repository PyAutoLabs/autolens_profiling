"""Unit tests for the point-solver stage map (``_point_solver_stage_map.py``).

The stage map is the instrument the point-source A100 bottleneck map
(``scripts/point_source_image/image_plane/gpu_bottleneck_map.py``,
autolens_profiling#350) is read through. These tests pin it on hand-built frames and
instructions whose right answer is known by construction:

1. **Caller context wins over the innermost physics frame.** A deflection is innermost in
   an ``autogalaxy`` mass profile whoever asked for it; the solver step, the magnification
   filter, beta* and the implicit-gradient JVP must still be told apart.
2. **The primal solve inside the ``custom_jvp`` stays a solver stage.** Its stack holds
   ``implicit_diff.py`` AND ``shape_solver._plane_grid``; it must not be swallowed by
   ``implicit_jvp``.
3. **The lattice / active split is by array size**, and a vmap batch axis cannot cross it.
4. **Every event lands in exactly one row** of ``stage_table`` and the kernel counts add up.

Only the last test imports the PyAuto* libraries (``library_ranges`` resolves the rule
line ranges from the installed source); it is skipped when they are not importable.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
for _d in (
    _ROOT / "scripts" / "misc",
    _ROOT / "scripts" / "point_source_image" / "image_plane",
):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import _point_solver_stage_map as sm  # noqa: E402
from likelihood_breakdown import xla_attribution as xa  # noqa: E402

_LENS = "/x/PyAutoLens/autolens/"
_ARRAY = "/x/PyAutoArray/autoarray/"
_GALAXY = "/x/PyAutoGalaxy/autogalaxy/"

#: Synthetic line ranges: each function occupies its own block of lines.
RANGES = {key: (100 * (i + 1), 100 * (i + 1) + 50) for i, key in enumerate(sm.LIBRARY_FUNCTIONS)}
RULES = sm.build_stage_map(RANGES)


def _line(key: str) -> int:
    return RANGES[key][0] + 10


def frame(path: str, line: int = 1, function: str = "f") -> xa.Frame:
    return xa.Frame(file=path, function=function, line=line)


MASS = frame(_GALAXY + "profiles/mass/total/isothermal.py", 200, "deflections_yx_2d_from")
PLANE_GRID = frame(_LENS + "point/solver/shape_solver.py", _line("plane_grid"), "_plane_grid")
MAGNIFICATION = frame(
    _LENS + "point/solver/shape_solver.py",
    _line("filter_low_magnification"),
    "_filter_low_magnification",
)
IMPLICIT = frame(_LENS + "point/solver/implicit_diff.py", 190, "solve_padded_jvp")
SOLVED = frame(_LENS + "point/fit/solved.py", 190, "_beta_hat")
POSITIONS = frame(_LENS + "point/fit/positions/image/abstract.py", 125, "model_data")
SHAPE = frame(_ARRAY + "structures/triangles/shape.py", 110, "_barycentric_contains")
STEP0 = frame(
    _ARRAY + "structures/triangles/array.py", _line("step0_point_mask"), "_step0_point_mask"
)
NEIGHBOURHOOD = frame(
    _ARRAY + "structures/triangles/coordinate_array.py",
    _line("coord_neighborhood"),
    "neighborhood",
)


def inst(name: str, frames, shape: str = "f64[20,2]", opcode: str = "fusion") -> xa.Instruction:
    return xa.Instruction(
        name=name, opcode=opcode, computation="main", shape=shape, frames=tuple(frames)
    )


def label_of(frames, shape="f64[20,2]") -> str:
    i = inst("x", frames, shape)
    return sm.classify(i, {"x": i}, RULES)


def test_solver_step_deflection_is_split_by_size():
    stack = [MASS, PLANE_GRID, POSITIONS]
    assert label_of(stack, "f64[11859,2]") == "deflections_lattice"
    assert label_of(stack, "f64[960,2]") == "deflections_active"
    # A vmap batch axis (<= 256) on an active-set array stays active.
    assert label_of(stack, "f64[256,960,2]") == "deflections_active"


def test_magnification_filter_and_beta_star_contexts():
    assert label_of([MASS, MAGNIFICATION, POSITIONS]) == "magnification_filter"
    assert label_of([MASS, SOLVED]) == "beta_star"


def test_implicit_jvp_only_when_no_primal_solver_context():
    # The jacfwd of the deflections at the solved images: implicit_diff, no solver step.
    assert label_of([MASS, IMPLICIT, POSITIONS]) == "implicit_jvp"
    # The primal solve traced inside the custom_jvp is still the solver step.
    assert label_of([MASS, PLANE_GRID, IMPLICIT, POSITIONS]) == "deflections_active"


def test_step0_containment_and_generic_containment():
    assert label_of([SHAPE, STEP0], "f64[23283]") == "step0_containment"
    assert label_of([STEP0], "pred[23283]") == "step0_containment"
    assert label_of([SHAPE], "pred[23283]") == "containment_lattice"
    assert label_of([SHAPE], "pred[320]") == "containment_active"


def test_bookkeeping_by_function_and_other():
    assert label_of([NEIGHBOURHOOD], "f64[80,2]") == "neighbourhood_active"
    assert label_of([frame("/x/numpy/whatever.py")]) == xa.OTHER
    assert label_of([]) == xa.OTHER
    assert label_of([POSITIONS]) == "chi_squared"


def test_every_event_lands_in_exactly_one_row():
    a = inst("a", [MASS, PLANE_GRID], "f64[11859,2]")
    b = inst("b", [SHAPE, STEP0], "pred[23283]")
    c = inst("c", [MASS, MAGNIFICATION])
    index = {"a": a, "b": b, "c": c}
    events = []
    t = 0
    calls = 3
    for _ in range(calls):
        for op, dur in (("a", 5000), ("b", 3000), (None, 1000), ("c", 2000), ("zz", 500)):
            name = "memset32" if op is None else f"kernel_{op}"
            events.append({"name": name, "start_ns": t, "dur_ns": dur, "hlo_op": op})
            t += dur + 700  # 0.7 us launch gap
    table = sm.stage_table(events, index, calls=calls, stage_map=RULES)
    assert table["call_split"].startswith("per-call")
    rows = table["per_stage"]
    assert rows["deflections_lattice"]["median_ms"] == pytest.approx(0.005)
    assert rows["step0_containment"]["median_ms"] == pytest.approx(0.003)
    assert rows["magnification_filter"]["median_ms"] == pytest.approx(0.002)
    assert rows[xa.MEMSET]["median_ms"] == pytest.approx(0.001)
    assert rows[xa.UNJOINED]["median_ms"] == pytest.approx(0.0005)
    # The partition: kernel counts and milliseconds add up to the events, per call.
    assert table["kernels_per_call"] == 5
    assert sum(r["kernels"] for r in rows.values()) == 5
    assert table["kernel_ms_per_call"] == pytest.approx(0.0115)
    assert sum(r["share_of_kernel_pct"] for r in rows.values()) == pytest.approx(100.0)
    gaps = table["inter_kernel_gap_us"]
    assert gaps["median"] == pytest.approx(0.7)
    assert table["device_busy_ms_per_call_median"] == pytest.approx(0.0115)


def test_gaps_survive_exact_ties():
    # Two thunks with the same (start, end) -- xla_attribution.idle_gaps compares the event
    # dicts here; the local helper must not.
    events = [
        {"name": "k", "start_ns": 0, "dur_ns": 10, "hlo_op": "a"},
        {"name": "k", "start_ns": 0, "dur_ns": 10, "hlo_op": "b"},
        {"name": "k", "start_ns": 30, "dur_ns": 10, "hlo_op": "c"},
    ]
    assert sm.inter_kernel_gaps_ms(events) == [pytest.approx(20 / 1e6)]


def test_library_ranges_resolve_against_installed_source():
    pytest.importorskip("autolens")
    ranges = sm.library_ranges()
    assert set(ranges) == set(sm.LIBRARY_FUNCTIONS)
    for key, (first, last) in ranges.items():
        assert 0 < first <= last, key
    rules = sm.build_stage_map(ranges)
    assert len(rules) > 20


def test_shape_dims_reads_pred_and_tuple_shapes():
    assert sm.shape_dims("pred[23283]{0}") == [23283]
    assert sm.shape_dims("(f64[20,2]{1,0}, s32[20]{0})") == [20, 2, 20]
    assert sm.shape_dims("f64[]") == []
