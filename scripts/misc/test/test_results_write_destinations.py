"""No script names a results path outside the mirrored tree (autolens_profiling#442, PR 3b).

Since the day-zero move ``results/`` holds only the mirrored tree: ``results/<dataset_type>/``
(one folder per ``scripts/<dataset_type>/``), ``results/lens/`` and ``results/misc/<group>/``.
The pre-#442 folders (``results/runtime/``, ``results/breakdown/``, ``results/simulators/``,
``results/notes/``, ``results/hazards/``, ...) keep only a pointer README and the map's
unresolved rows; a script that composed a path under one of them, or under any other
top-level folder (``results/latent/``, ``results/jit/``), would re-grow the legacy layout.

Heuristic (source text, no execution):

- **Python** (``scripts/**/*.py`` and ``_profile_cli.py``), parsed with ``ast``. Every path
  expression is flattened into its segments: ``/`` joins, ``<path>.joinpath(a, b)``,
  ``os.path.join(a, b)``, ``Path(a, b)`` / ``PurePosixPath(a, b)``, ``"a/b"`` constants split on
  ``/``, and a bare name bound earlier in the same module by a simple ``NAME = <path>``
  assignment (the variable-root alias ``R = ROOT / "results"; R / "runtime"``). A segment
  ``results`` that starts the path or follows an unknown root (a name, call or parameter --
  not another constant folder, so ``.../jax_profiling/results/jit`` is someone else's tree)
  must be followed by ``misc``, ``lens`` or a ``scripts/`` dataset folder, and a dataset folder
  by a variant that ``scripts/<dataset_type>/<variant>/`` also has. A string constant or
  f-string part holding ``results/<top>[/<variant>]`` at a path boundary (start, whitespace,
  quote, ``(``, ``=``, ``}/`` or ``)/``) is held to the same rule.
- Not followed: names bound in another module or by a non-trivial expression (a function
  parameter, ``x = a if c else b``, attribute assignments, loop targets), and dynamic segments
  (the check accepts what it cannot read). These need execution to resolve; the placement
  helpers and code review cover them.
- **Shell** (``hpc/**`` submit scripts and ``scripts/**/*.sh``): a non-comment line holding
  ``results/<top>`` at a path boundary (start, whitespace, quote, ``=``, ``ROOT/``, ``}/``).

Principled exclusions (keep this list explicit and small; prefer fixing a script):

- Docstrings and bare string statements, and comments (not in the AST; ``#`` lines in shell):
  they describe history, not a destination.
- Arguments of the placement / legacy-name helpers in ``_profile_cli`` (:data:`HELPERS`):
  ``resolve_output_paths(default_dir=...)``, ``place_output_path``, ``mirrored_output_path``,
  ``legacy_landing_dir``, ``locate_legacy_result``, ``legacy_result_files``,
  ``legacy_read_dirs``, ``legacy_result_dirs``. There a legacy folder is a *name* the helper
  places into, or resolves from, the mirrored tree.
- Assignments to a ``LEGACY_*`` constant (:data:`LEGACY_NAME`): a reader's legacy folder,
  passed only to those helpers (e.g. ``size_sweep_table.LEGACY_ROWS_DIR``).
- ``scripts/misc/test/``: fixtures build legacy layouts on purpose.
- :data:`EXCLUDED_FILES`: the migration tooling that reasons about the legacy layout itself,
  and ``baseline_readiness.py``, which validates the ``results/campaigns/`` evidence root that
  ``baseline/README.md`` reserves for a future producer.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

#: Top-level results folders the mirrored tree allows.
ALLOWED_TOPS = {"misc", "lens"} | {
    d.name for d in (ROOT / "scripts").iterdir() if d.is_dir() and not d.name.startswith("_")
}
HELPERS = {
    "resolve_output_paths",
    "place_output_path",
    "mirrored_output_path",
    "legacy_landing_dir",
    "locate_legacy_result",
    "legacy_result_files",
    "legacy_read_dirs",
    "legacy_result_dirs",
}
LEGACY_NAME = re.compile(r"^_?LEGACY_[A-Z0-9_]+$")
#: Tooling that reads or rewrites the legacy layout as its job (the day-zero map, its applier
#: and checker, the wiki checker that verifies the moved ledgers left nothing behind, the
#: legacy-series-id map the dashboard / README builders key on), plus one documented reader.
EXCLUDED_FILES = {
    "scripts/misc/tooling/build_migration_map.py",
    "scripts/misc/tooling/apply_migration_map.py",
    "scripts/misc/tooling/check_results_layout.py",
    "scripts/misc/tooling/check_wiki.py",
    "scripts/misc/tooling/build_series_id_map.py",
    # Reads the baseline campaign evidence root ``results/campaigns/<id>/`` that
    # ``baseline/README.md`` fixes for a future producer; no script writes there yet.
    "scripts/misc/tooling/baseline_readiness.py",
}
_PY_STRING = re.compile(r"(?:^/?|[\s\"'`(=]|\}/|\)/)results/([A-Za-z0-9_]+)(?:/([A-Za-z0-9_]+))?")
#: (dataset_type, variant) pairs ``scripts/`` has; a results/<dataset_type>/ path must name one.
VARIANTS = {
    (d.name, v.name)
    for d in (ROOT / "scripts").iterdir()
    if d.is_dir() and d.name not in ("misc", "lens") and not d.name.startswith("_")
    for v in d.iterdir()
    if v.is_dir() and not v.name.startswith("_")
}
#: Call names that build a path from their arguments (and, for joinpath, the receiver).
PATH_CALLS = {"Path", "PurePosixPath", "PurePath", "join", "joinpath"}
_SHELL = re.compile(r"(?:^|[\s\"'=]|ROOT\}?/|\}/)results/([A-Za-z0-9_]+)(?:/([A-Za-z0-9_]+))?")


def _bad(top: str) -> bool:
    return top not in ALLOWED_TOPS


def _violation(top: str, variant: str | None) -> str | None:
    """The offending ``results/...`` text, or None when the path is in the mirrored tree."""
    if _bad(top):
        return f"results/{top}"
    if top not in ("misc", "lens") and variant is not None and (top, variant) not in VARIANTS:
        return f"results/{top}/{variant}"
    return None


#: A segment whose value the matcher cannot read (a name, call, parameter, f-string field).
UNKNOWN = None


def _segments(node: ast.AST, aliases: dict[str, list]) -> list:
    """The path segments ``node`` builds (``UNKNOWN`` where a segment cannot be read)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [part for part in node.value.split("/") if part] or [UNKNOWN]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _segments(node.left, aliases) + _segments(node.right, aliases)
    if isinstance(node, ast.Name):
        return list(aliases.get(node.id, [UNKNOWN]))
    if isinstance(node, ast.Call) and _call_name(node) in PATH_CALLS:
        out = []
        if _call_name(node) == "joinpath" and isinstance(node.func, ast.Attribute):
            out = _segments(node.func.value, aliases)
        for arg in node.args:
            out += _segments(arg, aliases)
        return out
    return [UNKNOWN]


def _path_violation(segs: list) -> str | None:
    for i, seg in enumerate(segs):
        if seg != "results" or i + 1 >= len(segs) or segs[i + 1] is UNKNOWN:
            continue
        if i > 0 and segs[i - 1] is not UNKNOWN:
            continue  # results/ inside another constant folder: someone else's tree
        variant = segs[i + 2] if i + 2 < len(segs) else None
        found = _violation(segs[i + 1], variant)
        if found:
            return found
    return None


def _aliases(tree: ast.AST) -> dict[str, list]:
    """Segments of every name bound by a simple same-module ``NAME = <path>`` assignment.

    Bound in source order; a name bound twice keeps its last value. ``LEGACY_*`` names are
    declared legacy names (an exclusion) and are not followed.
    """
    aliases: dict[str, list] = {}
    assigns = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for node in sorted(assigns, key=lambda n: (n.lineno, n.col_offset)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name) or node.value is None:
            continue
        name = targets[0].id
        if LEGACY_NAME.match(name):
            continue
        segs = _segments(node.value, aliases)
        if any(seg is not UNKNOWN for seg in segs):
            aliases[name] = segs
        else:
            aliases.pop(name, None)
    return aliases


def _call_name(node: ast.Call) -> str:
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
    return name.lstrip("_")


def _excluded_nodes(tree: ast.AST) -> set[int]:
    """ids of every node inside a docstring, a helper call's arguments or a LEGACY_* value."""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            roots.append(node)  # docstrings and bare string statements
        elif isinstance(node, ast.Call) and _call_name(node) in HELPERS:
            roots.extend(node.args)
            roots.extend(k.value for k in node.keywords)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and LEGACY_NAME.match(t.id) for t in targets):
                if node.value is not None:
                    roots.append(node.value)
    return {id(n) for root in roots for n in ast.walk(root)}


def python_hits(source: str) -> list[tuple[int, str]]:
    """``(line, offending text)`` of every results path outside the mirrored tree."""
    tree = ast.parse(source)
    skip = _excluded_nodes(tree)
    aliases = _aliases(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        is_join = isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        if is_join or (isinstance(node, ast.Call) and _call_name(node) in PATH_CALLS):
            found = _path_violation(_segments(node, aliases))
            if found:
                hits.append((node.lineno, found))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in _PY_STRING.finditer(node.value):
                found = _violation(m.group(1), m.group(2))
                if found:
                    hits.append((node.lineno, found))
    # A join nested in a longer join reports once per line.
    return sorted(set(hits))


def shell_hits(text: str) -> list[tuple[int, str]]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in _SHELL.finditer(line):
            found = _violation(m.group(1), m.group(2))
            if found:
                hits.append((i, found))
    return hits


def _python_files() -> list[Path]:
    files = [ROOT / "_profile_cli.py"]
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("scripts/misc/test/") or rel in EXCLUDED_FILES:
            continue
        files.append(path)
    return files


def _shell_files() -> list[Path]:
    files = sorted((ROOT / "scripts").rglob("*.sh"))
    for path in sorted((ROOT / "hpc").rglob("*")):
        if not path.is_file() or {"output", "error"} & set(path.relative_to(ROOT).parts):
            continue
        if path.name.startswith("submit_") or path.suffix == ".sh" or path.name == "sync":
            files.append(path)
    return files


def test_the_matcher_catches_a_legacy_write_and_spares_the_exclusions():
    assert python_hits('out = ROOT / "results" / "breakdown" / "x.json"\n') == [
        (1, "results/breakdown")
    ]
    assert python_hits('p = ROOT / "results/simulators/x.json"\n') == [(1, "results/simulators")]
    assert python_hits('p = ROOT / "results" / "latent"\n') == [(1, "results/latent")]
    assert python_hits('p = f"{root}/results/notes/x.md"\n') == [(1, "results/notes")]
    # the forms the first matcher missed (PR 3b review finding 3)
    assert python_hits('R = ROOT / "results"\nout = R / "runtime" / "x.json"\n') == [
        (2, "results/runtime")
    ]
    assert python_hits('R = ROOT / "results"\nRT = R / "runtime"\n') == [(2, "results/runtime")]
    assert python_hits('p = ROOT.joinpath("results", "runtime", "x.json")\n') == [
        (1, "results/runtime")
    ]
    assert python_hits('p = os.path.join(root, "results", "breakdown")\n') == [
        (1, "results/breakdown")
    ]
    assert python_hits('p = Path("results", "runtime")\n') == [(1, "results/runtime")]
    assert python_hits('p = ROOT / "results" / "imaging" / "pixelization" / "x.json"\n') == [
        (1, "results/imaging/pixelization")
    ]
    assert python_hits('p = "results/interferometer/nonsense/x.json"\n') == [
        (1, "results/interferometer/nonsense")
    ]
    spared = (
        '"""Rows were written to results/breakdown/imaging/ before #442."""\n'
        "# results/runtime/ was the old home\n"
        'a, b = resolve_output_paths(cli, default_dir=ROOT / "results" / "runtime" / "x", '
        'default_basename="x")\n'
        'p = _place_output_path(ROOT / "results" / "breakdown" / "x.json", cli, "x")\n'
        'LEGACY_ROWS_DIR = ROOT / "results" / "breakdown" / "imaging"\n'
        'q = ROOT / "results" / "misc" / "likelihood_breakdown" / "imaging"\n'
        'r = ROOT / "results" / "imaging" / "delaunay"\n'
        's = "autolens_workspace_developer/jax_profiling/results/jit"\n'
        't = WS / "jax_profiling" / "results" / "jit"\n'
        'R = ROOT / "results"\nu = R / "misc" / "hazards"\nv = R.joinpath("imaging", "mge")\n'
        'w = Path(ROOT, "results", dataset_type, variant)\n'
    )
    assert python_hits(spared) == []
    assert shell_hits("    --output-dir $AP_ROOT/results/breakdown/interferometer\n") == [
        (1, "results/breakdown")
    ]
    assert shell_hits('RESULTS_DIR="${AP_ROOT}/results/runtime/x"\n') == [(1, "results/runtime")]
    assert shell_hits("# landed in results/breakdown/imaging/\nX=$AP_ROOT/results/misc/a\n") == []
    assert shell_hits("X=$AP_ROOT/results/imaging/pixelization/a\n") == [
        (1, "results/imaging/pixelization")
    ]


def test_no_script_writes_outside_the_mirrored_tree():
    found = []
    for path in _python_files():
        for line, text in python_hits(path.read_text()):
            found.append(f"{path.relative_to(ROOT)}:{line}: {text}")
    for path in _shell_files():
        for line, text in shell_hits(path.read_text(errors="replace")):
            found.append(f"{path.relative_to(ROOT)}:{line}: {text}")
    assert not found, (
        "results paths outside the mirrored tree (route them through "
        "_profile_cli.resolve_output_paths / place_output_path, or write results/misc/<group>/):"
        "\n" + "\n".join(found)
    )


def test_the_excluded_files_exist():
    missing = sorted(p for p in EXCLUDED_FILES if not (ROOT / p).is_file())
    assert not missing, f"stale EXCLUDED_FILES entries: {missing}"
