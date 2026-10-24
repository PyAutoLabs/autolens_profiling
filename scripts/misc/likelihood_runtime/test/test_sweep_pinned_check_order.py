"""Every sweep-driven leaf records its pinned check after its last row write, before fit.png.

``sweep.row_completed_since`` treats a row as finished once it carries ``pinned_drift``
(autolens_profiling#453), so a sweep timeout during the ``fit.png`` render keeps the row. That
holds only while each leaf in ``sweep_layout.CELLS`` calls ``record_pinned_check`` after its
last ``write_text`` of the row and before ``render_fit_png``. The leaves run a profile on
import, so they are read as source (#454 item 6).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

_MISC = Path(__file__).resolve().parents[2]
ROOT = _MISC.parents[1]
for _p in (ROOT, _MISC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Loaded by file path under its own name (see test_sweep_timeout.py).
_spec = importlib.util.spec_from_file_location(
    "sweep_layout_pinned_order", _MISC / "likelihood_runtime" / "sweep_layout.py"
)
layout = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = layout
_spec.loader.exec_module(layout)

from _script_routes import runtime_path  # noqa: E402

TRACKED = ("write_text", "record_pinned_check", "render_fit_png")


def _module_level_calls(source: str) -> list[tuple[int, str]]:
    """``(line, name)`` of each tracked call the module body runs (function bodies excluded)."""
    found = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in TRACKED:
                found.append((node.lineno, name))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return sorted(found)


def order_problem(source: str) -> str | None:
    """Why the leaf breaks the rule, or None when it records the check in order."""
    calls = _module_level_calls(source)
    writes = [line for line, name in calls if name == "write_text"]
    checks = [line for line, name in calls if name == "record_pinned_check"]
    renders = [line for line, name in calls if name == "render_fit_png"]
    if not writes:
        return "no row write_text"
    after = [line for line in checks if line > writes[-1]]
    if not after:
        return f"no record_pinned_check after the last row write (line {writes[-1]})"
    if renders and not any(line < min(renders) for line in after):
        return f"record_pinned_check comes after render_fit_png (line {min(renders)})"
    return None


def test_every_sweep_leaf_records_its_pinned_check_after_the_row_write():
    problems = []
    for cls, model, _instruments in layout.CELLS:
        script = runtime_path(cls, model, ROOT)
        problem = order_problem(script.read_text())
        if problem:
            problems.append(f"{script.relative_to(ROOT)}: {problem}")
    assert not problems, "\n".join(problems)
    assert len(layout.CELLS) >= 8


def test_the_order_rule_catches_each_wrong_order():
    good = 'p.write_text("{}")\nrecord_pinned_check(p, e, d)\nrender_fit_png(f, p.parent)\n'
    assert order_problem(good) is None
    late = 'p.write_text("{}")\nrender_fit_png(f, p.parent)\nrecord_pinned_check(p, e, d)\n'
    assert "after render_fit_png" in order_problem(late)
    early = 'record_pinned_check(p, e, d)\np.write_text("{}")\nrender_fit_png(f, p.parent)\n'
    assert "after the last row write" in order_problem(early)
    # A call inside a function body is not the module's run order.
    hidden = 'def f():\n    record_pinned_check(p, e, d)\np.write_text("{}")\nf()\n'
    assert "after the last row write" in order_problem(hidden)
    assert order_problem("record_pinned_check(p, e, d)\n") == "no row write_text"
