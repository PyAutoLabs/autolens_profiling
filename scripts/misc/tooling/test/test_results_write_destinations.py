"""No script names a results path outside the mirrored tree (autolens_profiling#442, PR 3b).

Since the day-zero move ``results/`` holds only the mirrored tree: ``results/<dataset_type>/``
(one folder per ``scripts/<dataset_type>/``), ``results/lens/`` and ``results/misc/<group>/``.
The pre-#442 folders (``results/runtime/``, ``results/breakdown/``, ``results/simulators/``,
``results/notes/``, ``results/hazards/``, ...) keep only a pointer README and the map's
unresolved rows; a script that composed a path under one of them, or under any other
top-level folder (``results/latent/``, ``results/jit/``), would re-grow the legacy layout.

Heuristic (source text, no execution):

- **Python** (``scripts/**/*.py`` and ``_profile_cli.py``), parsed with ``ast``: a ``/`` path
  join whose operands hold the constant ``"results"`` immediately followed by a constant
  top-level folder, or a string constant / f-string part holding ``results/<top>`` at a path
  boundary (start of the string, whitespace, quote, ``(``, ``=``, ``}/`` or ``)/``), where ``<top>`` is not ``misc``, ``lens`` or a ``scripts/`` dataset-type folder.
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
- ``scripts/misc/<group>/test/``: fixtures build legacy layouts on purpose.
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
_PY_STRING = re.compile(r"(?:^/?|[\s\"'`(=]|\}/|\)/)results/([A-Za-z0-9_]+)")
_SHELL = re.compile(r"(?:^|[\s\"'=]|ROOT\}?/|\}/)results/([A-Za-z0-9_]+)")


def _bad(top: str) -> bool:
    return top not in ALLOWED_TOPS


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


def _div_operands(node: ast.BinOp) -> list[ast.AST]:
    out: list[ast.AST] = []
    for side in (node.left, node.right):
        if isinstance(side, ast.BinOp) and isinstance(side.op, ast.Div):
            out.extend(_div_operands(side))
        else:
            out.append(side)
    return out


def python_hits(source: str) -> list[tuple[int, str]]:
    """``(line, offending text)`` of every results path outside the mirrored tree."""
    tree = ast.parse(source)
    skip = _excluded_nodes(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            ops = _div_operands(node)
            for a, b in zip(ops, ops[1:]):
                if (
                    isinstance(a, ast.Constant)
                    and a.value == "results"
                    and isinstance(b, ast.Constant)
                    and isinstance(b.value, str)
                    and _bad(b.value.split("/")[0])
                ):
                    hits.append((node.lineno, f'"results" / "{b.value}"'))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in _PY_STRING.finditer(node.value):
                if _bad(m.group(1)):
                    hits.append((node.lineno, f"results/{m.group(1)}"))
    return sorted(set(hits))


def shell_hits(text: str) -> list[tuple[int, str]]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in _SHELL.finditer(line):
            if _bad(m.group(1)):
                hits.append((i, f"results/{m.group(1)}"))
    return hits


def _python_files() -> list[Path]:
    files = [ROOT / "_profile_cli.py"]
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if re.match(r"scripts/misc/[^/]+/test/", rel) or rel in EXCLUDED_FILES:
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
        (1, '"results" / "breakdown"')
    ]
    assert python_hits('p = ROOT / "results/simulators/x.json"\n') == [(1, "results/simulators")]
    assert python_hits('p = ROOT / "results" / "latent"\n') == [(1, '"results" / "latent"')]
    assert python_hits('p = f"{root}/results/notes/x.md"\n') == [(1, "results/notes")]
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
    )
    assert python_hits(spared) == []
    assert shell_hits("    --output-dir $AP_ROOT/results/breakdown/interferometer\n") == [
        (1, "results/breakdown")
    ]
    assert shell_hits('RESULTS_DIR="${AP_ROOT}/results/runtime/x"\n') == [(1, "results/runtime")]
    assert shell_hits("# landed in results/breakdown/imaging/\nX=$AP_ROOT/results/misc/a\n") == []


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
