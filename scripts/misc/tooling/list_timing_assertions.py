"""List the timing assertions and timing-gate constants in this repo (read-only).

The timing-noise audit (autolens_profiling#362) keeps one inventory of every
place a measured time is compared against a cutoff:
``results/notes/timing_noise_audit_2026_10.md``. This lister finds those places
mechanically, so the inventory cannot silently fall behind the code.

It walks every ``*.py`` under ``scripts/`` plus the repo-root helpers with
``ast`` (nothing is imported or executed) and reports two kinds of site:

``compare``
    An ordering comparison (``<``, ``<=``, ``>``, ``>=``) whose source text names
    a timing quantity (ms, seconds, wall, elapsed, ratio, speedup, overhead,
    median, load average, headroom, budget, saving, ...) and whose other side is a number or an UPPER_CASE constant.
    Comparisons against a literal ``0`` or a tolerance of at most ``1e-9`` are
    tagged ``validity`` (a sign or identity check, not a noise cutoff); the rest
    are tagged ``threshold``.
``constant``
    A module-level UPPER_CASE numeric assignment whose name says it is a timing
    budget, ratio, repeat count, warm-up or bootstrap setting.

Each site has a stable key: ``<path>::<function>`` for a comparison (one key
per enclosing function; ``<module>`` for module-level code) and
``<path>::<NAME>`` for a constant. Line numbers are printed but never keyed, so
an unrelated edit above a site does not churn the inventory.

The heuristic is deliberately broad: it over-reports (a formatting branch on
``seconds < 1`` is listed) rather than under-reports. The inventory note either
gives each key a row or lists it under "Lister hits that are not timing gates"
with the reason.

Run from the repo root::

    python scripts/misc/tooling/list_timing_assertions.py           # table
    python scripts/misc/tooling/list_timing_assertions.py --all     # + validity hits
    python scripts/misc/tooling/list_timing_assertions.py --check   # fail on drift

``--check`` exits non-zero when a ``threshold`` or ``constant`` key is missing
from the note, so a new timing gate has to be inventoried in the same PR that
adds it. Validity keys are not required. Pure stdlib; no PyAuto* imports.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
NOTE = ROOT / "results" / "notes" / "timing_noise_audit_2026_10.md"
SELF = Path(__file__).resolve()

#: Source text that marks a comparison as being about a measured time.
TIMING_TOKEN = re.compile(
    r"(?i)(_ms\b|\bms_|_s\b|_s\[|seconds|wall|elapsed|latency|runtime|per_call|"
    r"speedup|slowdown|overhead|ratio|median|_time\b|\btime_|duration|_over_|loadavg|"
    r"headroom|budget|saving|saved)"
)

#: Module-level constant names that declare a timing budget or protocol setting.
CONSTANT_NAME = re.compile(
    r"^(?:[A-Z0-9_]*(?:OVERHEAD|SPEEDUP|DRIFT|LEVER|SAVED|WARMUP|LOADAVG|HEADROOM|"
    r"STEP_SUM)[A-Z0-9_]*"
    r"|[A-Z0-9_]*(?:BUDGET_S|BUDGET_MS|_MS|_SECONDS|RATE_TOLERANCE)"
    r"|(?:[A-Z0-9_]+_)?(?:N_REPEATS|REPEATS|N_WARM|N_STEADY|N_TIMED|N_ROUNDS|N_CALLS|"
    r"MIN_BLOCKS[A-Z0-9_]*|MIN_STEADY_WARM|BOOTSTRAP_SAMPLES|GO_MIN_[A-Z_]+))$"
)

#: A comparison against a literal at or below this magnitude is a validity check.
VALIDITY_EPS = 1.0e-9


@dataclass(frozen=True)
class Site:
    kind: str  # "threshold" | "validity" | "constant"
    key: str
    line: int
    text: str


def _is_number(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp):
        return _is_number(node.operand)
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    )


def _number(node: ast.AST) -> float | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return None if value is None else -value
    if _is_number(node):
        return float(node.value)
    return None


def _is_cutoff(node: ast.AST) -> bool:
    """A number, an UPPER_CASE name/attribute, or arithmetic involving one."""
    if _is_number(node):
        return True
    if isinstance(node, ast.Name):
        return node.id.isupper()
    if isinstance(node, ast.Attribute):
        return node.attr.isupper()
    if isinstance(node, ast.BinOp):
        return _is_cutoff(node.left) or _is_cutoff(node.right)
    return False


def _is_validity(node: ast.Compare) -> bool:
    """Every literal side is 0 or a tolerance at most VALIDITY_EPS (``x + 1e-9``)."""
    literals: list[float] = []
    for side in (node.left, *node.comparators):
        value = _number(side)
        if value is not None:
            literals.append(value)
        elif isinstance(side, ast.BinOp):
            for part in (side.left, side.right):
                inner = _number(part)
                if inner is not None:
                    literals.append(inner)
    if not literals:
        return False
    return all(abs(v) <= VALIDITY_EPS for v in literals)


def source_files(root: Path = ROOT) -> list[Path]:
    files = sorted(root.glob("scripts/**/*.py")) + sorted(root.glob("*.py"))
    return [p for p in files if p.resolve() != SELF]


def scan_file(path: Path, root: Path = ROOT) -> list[Site]:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    rel = path.relative_to(root).as_posix()
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def enclosing(node: ast.AST) -> str:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return node.name
        return "<module>"

    sites: list[Site] = []
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None or not _is_cutoff(value):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and CONSTANT_NAME.match(target.id):
                    text = (ast.get_source_segment(source, node) or "").splitlines()[0]
                    sites.append(Site("constant", f"{rel}::{target.id}", node.lineno, text))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE) for op in node.ops):
            continue
        sides = [node.left, *node.comparators]
        if not any(_is_cutoff(side) for side in sides):
            continue
        text = " ".join((ast.get_source_segment(source, node) or "").split())
        if not TIMING_TOKEN.search(text):
            continue
        kind = "validity" if _is_validity(node) else "threshold"
        sites.append(Site(kind, f"{rel}::{enclosing(node)}", node.lineno, text[:100]))
    return sites


def scan(root: Path = ROOT) -> list[Site]:
    sites: list[Site] = []
    for path in source_files(root):
        sites.extend(scan_file(path, root))
    return sorted(sites, key=lambda s: (s.key, s.line))


def required_keys(sites: list[Site]) -> list[str]:
    """Keys the inventory note must mention: every threshold and constant key."""
    return sorted({s.key for s in sites if s.kind != "validity"})


def missing_from_note(sites: list[Site], note_text: str) -> list[str]:
    return [key for key in required_keys(sites) if f"`{key}`" not in note_text]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="include validity comparisons")
    parser.add_argument(
        "--check", action="store_true", help="fail if the inventory note misses a key"
    )
    args = parser.parse_args(argv)

    sites = scan()
    if args.check:
        if not NOTE.exists():
            print(f"FAIL: inventory note missing: {NOTE.relative_to(ROOT)}")
            return 1
        missing = missing_from_note(sites, NOTE.read_text())
        if missing:
            print(
                f"FAIL: {len(missing)} timing site(s) found in code but absent from "
                f"{NOTE.relative_to(ROOT)} (add a row, or list the key under "
                f"'Lister hits that are not timing gates' with the reason):"
            )
            for key in missing:
                print(f"  `{key}`")
            return 1
        print(f"OK: {len(required_keys(sites))} timing keys, all inventoried.")
        return 0

    shown = [s for s in sites if args.all or s.kind != "validity"]
    width = max((len(s.key) for s in shown), default=10)
    print(f"{'kind':<9}  {'key':<{width}}  {'line':>5}  text")
    for s in shown:
        print(f"{s.kind:<9}  {s.key:<{width}}  {s.line:>5}  {s.text}")
    counts = {k: sum(s.kind == k for s in sites) for k in ("threshold", "constant", "validity")}
    print(
        f"\n{counts['threshold']} threshold comparison(s), {counts['constant']} constant(s), "
        f"{counts['validity']} validity comparison(s); "
        f"{len(required_keys(sites))} key(s) the inventory must cover."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
