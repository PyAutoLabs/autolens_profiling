"""Apply the day-zero migration map with ``git mv`` (stdlib only).

Run from the repo root::

    python scripts/misc/tooling/apply_migration_map.py            # dry run: count pending moves
    python scripts/misc/tooling/apply_migration_map.py --apply    # git mv every pending entry
    python scripts/misc/tooling/apply_migration_map.py --rewrite-refs  # repoint path references

Reads ``catalogue/migration_day_zero.json`` (written by ``build_migration_map.py``) and,
for every entry whose ``new`` differs from ``old``, runs ``git mv old new`` after checking
that the file's sha256 still matches the map. Content is never touched: a sha mismatch
aborts before anything moves. ``unresolved`` rows are not in ``entries`` and stay put.

Re-runnable: an entry whose ``old`` is gone and whose ``new`` is tracked is already
applied and is skipped, so the map can be regenerated (e.g. after a ledger PR merges)
and re-applied on a rebased branch.

``--rewrite-refs`` repoints references to moved files outside the result trees, from the
map alone: every literal repo-relative old path (``results/...``, ``wiki/campaigns/...``)
in a tracked text file becomes its new path, and every relative markdown link is
re-resolved from where its page used to live to where its target lives now. Result files
(``results/**``, ``archive/**``), generated outputs (``dashboard/``, ``wiki/setups/``) and
the map's own generator/tests are never edited. Idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAP_PATH = "catalogue/migration_day_zero.json"


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def tracked(root: Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout
    return {p for p in out.decode().split("\0") if p}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(root: Path, doc: dict) -> tuple[list[tuple[str, str]], int, list[str]]:
    """Return (pending moves, already-applied count, errors)."""
    files = tracked(root)
    pending, done, errors = [], 0, []
    for entry in doc["entries"]:
        old, new = entry["old"], entry["new"]
        if new == old:
            continue
        if new in files and (old not in files or sha256(root / old) != entry["sha256"]):
            done += 1  # applied; a file now at ``old`` is a new one (a pointer README)
            continue
        if old not in files:
            errors.append(f"{old}: not tracked and {new} absent")
            continue
        if new in files or (root / new).exists():
            errors.append(f"{old}: destination {new} already exists")
            continue
        if sha256(root / old) != entry["sha256"]:
            errors.append(f"{old}: sha256 differs from the map (regenerate the map)")
            continue
        pending.append((old, new))
    return pending, done, errors


#: Never rewritten: result bytes, generated outputs, and the day-zero map's own tooling.
#: Tests are excluded too: their literal ``results/...`` paths are mostly fixture trees in
#: the legacy layout; reads of committed rows name the mirrored path (#442 PR 3).
REWRITE_SKIP_PREFIXES = (
    "results/",
    "archive/",
    "dashboard/",
    "wiki/setups/",
    "scripts/misc/test/",
)
REWRITE_KEEP = ("results/README.md", "results/provenance_grandfathered.txt")
REWRITE_SKIP = (
    MAP_PATH,
    "scripts/misc/tooling/build_migration_map.py",
    "scripts/misc/tooling/apply_migration_map.py",
)
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".yml", ".yaml", ".conf", ".txt", ".toml", ""}
PATH_CHARS = r"A-Za-z0-9_.+\-"
LINK_RE = re.compile(r"(\]\()([^)\s]+)(\))|^(\[[^\]]+\]:\s+)(\S+)", re.MULTILINE)


def rewrite_targets(files: set[str]) -> list[str]:
    out = []
    for path in sorted(files):
        if path in REWRITE_SKIP:
            continue
        if path.startswith(REWRITE_SKIP_PREFIXES) and path not in REWRITE_KEEP:
            continue
        if Path(path).suffix in TEXT_SUFFIXES:
            out.append(path)
    return out


#: Where a link to an emptied results folder points: the redirect table.
REDIRECT_TABLE = "results/README.md#day-zero-move-redirects"


def dir_redirect(resolved: str, moves: dict, scripts: dict) -> str:
    """New target for a link whose old target is gone and is not a moved file.

    A legacy script route -> its canonical script; an emptied results folder -> the one
    folder all its files moved to (``results/notes`` -> ``wiki/campaigns``), else the
    redirect table in ``results/README.md``. Anything else is left as it was.
    """
    if resolved in scripts:
        return scripts[resolved]
    if resolved == "results/notes":
        return "wiki/campaigns/"
    news = [new for old, new in moves.items() if old.startswith(resolved + "/")]
    if not news:
        return resolved
    common = os.path.commonpath(news)
    if common.count("/") >= 2:
        return common + "/"
    return REDIRECT_TABLE


def _relink(text: str, page: str, old_page: str, moves: dict, root: Path, scripts=None) -> str:
    """Re-resolve relative markdown links of ``page`` (formerly at ``old_page``)."""

    scripts = scripts or {}

    def fix(target: str) -> str:
        if re.match(r"^[a-z]+:", target) or target.startswith(("#", "/", "<")):
            return target
        bare, hash_, anchor = target.partition("#")
        if not bare:
            return target
        here = os.path.normpath(os.path.join(os.path.dirname(page), bare))
        if not here.startswith(".."):
            if (root / here).exists():
                return target  # already resolves from where the page lives now (idempotent)
            moved = moves.get(here) or dir_redirect(here, moves, scripts)
            if moved != here:
                return relative(moved, bare) + ("" if "#" in moved else hash_ + anchor)
        resolved = os.path.normpath(os.path.join(os.path.dirname(old_page), bare))
        if resolved.startswith(".."):
            return target
        new = moves.get(resolved, resolved)
        if new == resolved and not (root / resolved).exists():
            new = dir_redirect(resolved, moves, scripts)
        if new == resolved and page == old_page:
            return target
        return relative(new, bare) + ("" if "#" in new else hash_ + anchor)

    def relative(new: str, bare: str) -> str:
        new, hash_, new_anchor = new.partition("#")
        trailing = "/" if new.endswith("/") or (bare.endswith("/") and not hash_) else ""
        rel = os.path.relpath(new.rstrip("/"), os.path.dirname(page) or ".")
        return rel + trailing + hash_ + new_anchor

    def sub(m: re.Match) -> str:
        if m.group(1):
            return m.group(1) + fix(m.group(2)) + m.group(3)
        return m.group(4) + fix(m.group(5))

    return LINK_RE.sub(sub, text)


def rewrite_refs(root: Path, doc: dict) -> list[str]:
    """Rewrite references to moved files; returns the files changed."""
    moves = {e["old"]: e["new"] for e in doc["entries"] if e["new"] != e["old"]}
    inverse = {new: old for old, new in moves.items()}
    pattern = re.compile(
        rf"(?<![{PATH_CHARS}])("
        + "|".join(re.escape(k) for k in sorted(moves, key=len, reverse=True))
        + rf")(?![{PATH_CHARS}/]|\.[A-Za-z0-9])"
    )
    routes_path = root / "catalogue/script_routes.json"
    scripts = {}
    if routes_path.exists():
        routes = json.loads(routes_path.read_text())["routes"]
        scripts = {row["legacy"]: row["path"] for row in routes}
    changed = []
    for path in rewrite_targets(tracked(root)):
        try:
            text = (root / path).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        new = text
        if path.endswith(".md"):
            new = _relink(new, path, inverse.get(path, path), moves, root, scripts)
        new = pattern.sub(lambda m: moves[m.group(1)], new)
        if new != text:
            (root / path).write_text(new)
            changed.append(path)
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="run git mv (default: dry run)")
    parser.add_argument(
        "--rewrite-refs", action="store_true", help="repoint references to moved files"
    )
    args = parser.parse_args(argv)

    root = _root()
    doc = json.loads((root / MAP_PATH).read_text())
    if args.rewrite_refs:
        changed = rewrite_refs(root, doc)
        for path in changed:
            print(f"  rewrote {path}")
        print(f"rewrote references in {len(changed)} files")
        return 0
    pending, done, errors = plan(root, doc)
    if errors:
        print(f"{len(errors)} map entries cannot be applied:", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"{len(pending)} pending, {done} already applied, "
        f"{len(doc.get('unresolved', []))} unresolved left in place"
    )
    if not args.apply:
        return 0
    for old, new in pending:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", old, new], cwd=root, check=True)
    print(f"moved {len(pending)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
