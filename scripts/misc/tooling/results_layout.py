"""The results tree after the day-zero move: legacy view for readers, placement for writers.

The move-only PR of autolens_profiling#442 applied ``catalogue/migration_day_zero.json``
with ``git mv``: ``results/{runtime,breakdown,...}`` became the mirrored tree
``results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>/...``, dead
experiments went to ``archive/`` and the ledgers to ``wiki/campaigns/``.

Two readers still parse the legacy file-name grammar (``<cell>_<config>[_sparse].json``):
``build_dashboard.py`` (its trend series are keyed by legacy section/cell/config) and
``build_readme.py`` (its README tables). They see every moved file under its **legacy
identity** -- the old path the map records -- while reading and linking the file where it now
lives. Every other reader, the catalogue builder and the tests included, reads the mirrored
tree natively (#442 PR 3); this module goes when those two are re-keyed on the v3 identity. A file the map does not know (a fixture, an
unresolved row left in place, a run written after the move) is its own legacy identity.

Writers place a new result with :func:`destination` (the map generator's classification,
imported, not duplicated). Pure stdlib.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from functools import lru_cache
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_migration_map as _mm  # noqa: E402

MAP_PATH = _mm.MAP_PATH
#: Trees that hold moved result files (archive/ keeps dead experiments readable).
RESULT_TREES = ("results", "archive")


@lru_cache(maxsize=8)
def _maps(root: str) -> tuple[dict, dict]:
    path = Path(root) / MAP_PATH
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}, {}
    moves = {e["old"]: e["new"] for e in doc.get("entries", []) if e["new"] != e["old"]}
    return moves, {new: old for old, new in moves.items()}


def moves(root: Path) -> dict:
    """{old repo-relative path: new path} for every moved file."""
    return _maps(str(Path(root).resolve()))[0]


def legacy_of(root: Path, rel: str) -> str:
    """The pre-move path of repo-relative ``rel`` (itself when the map does not move it)."""
    return _maps(str(Path(root).resolve()))[1].get(rel, rel)


def current_of(root: Path, legacy: str) -> str:
    """Where the file once at repo-relative ``legacy`` lives now."""
    return _maps(str(Path(root).resolve()))[0].get(legacy, legacy)


def iter_legacy(root: Path, prefix: str, pattern: str = "*") -> list[tuple[Path, PurePosixPath]]:
    """``(actual path, legacy path)`` of every file whose legacy path lies under ``prefix``.

    The legacy-tree equivalent of ``sorted((root / prefix).rglob(pattern))``: same order
    (sorted by legacy path), file names matched against ``pattern`` by their legacy name.
    """
    root = Path(root)
    prefix = prefix.rstrip("/")
    found = {}
    for top in RESULT_TREES:
        base = root / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if rel in moves(root):
                continue  # a new file at a moved file's old path (a pointer README)
            legacy = PurePosixPath(legacy_of(root, rel))
            if not (legacy.as_posix() == prefix or legacy.as_posix().startswith(prefix + "/")):
                continue
            if fnmatch.fnmatchcase(legacy.name, pattern):
                found[legacy] = p
    return [(found[k], k) for k in sorted(found)]


def _split(path: Path) -> tuple[Path, str] | None:
    """(repo root, repo-relative path) for a path inside a ``results/`` tree, else None."""
    path = Path(path).resolve()
    parts = path.parts
    if "results" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("results")
    return Path(*parts[:i]), "/".join(parts[i:])


def iter_legacy_dir(
    directory: Path, pattern: str = "*", recursive: bool = True
) -> list[tuple[Path, PurePosixPath]]:
    """``iter_legacy`` for a legacy directory given as a path: ``(actual path, path relative
    to ``directory`` in legacy terms)``. ``directory`` need not exist any more. Outside a
    ``results/`` tree it is a plain sorted ``rglob`` (``glob`` when not ``recursive``)."""
    split = _split(directory)
    if split is None:
        directory = Path(directory)
        if not directory.is_dir():
            return []
        found = directory.rglob(pattern) if recursive else directory.glob(pattern)
        return [
            (p, PurePosixPath(p.relative_to(directory).as_posix()))
            for p in sorted(found)
            if p.is_file()
        ]
    root, prefix = split
    out = [(p, legacy.relative_to(prefix)) for p, legacy in iter_legacy(root, prefix, pattern)]
    return out if recursive else [(p, rel) for p, rel in out if len(rel.parts) == 1]


def legacy_glob(directory: Path, pattern: str = "*") -> list[Path]:
    """``sorted(directory.glob(pattern))`` over a legacy folder: where those files live now."""
    return [p for p, _ in iter_legacy_dir(directory, pattern, recursive=False)]


def current(path: Path) -> Path:
    """Where the file once at ``path`` (absolute, in a legacy ``results/`` folder) lives now."""
    split = _split(path)
    if split is None:
        return Path(path)
    root, rel = split
    return root / current_of(root, rel)


def destination(root: Path, legacy: str, doc: dict | None = None) -> tuple[str, str]:
    """(new repo-relative path, kind) for a result a writer would have put at ``legacy``."""
    return _mm.destination(Path(root), legacy, doc)
