"""Gate: the profiling research wiki (`wiki/`) cannot silently drift from the ledgers.

Run from the repo root::

    python scripts/misc/tooling/check_wiki.py           # report
    python scripts/misc/tooling/check_wiki.py --check   # exit 1 on any failure

Checks
------

(a) every ``results/notes/*.md`` ledger is linked (by relative path) from at least
    one markdown file under ``wiki/``;
(b) every ``wiki/campaigns/*.md`` page except ``_template.md`` has a row in
    ``wiki/index.md`` (matched by link target) and carries all ten header labels
    of the page contract (see ``wiki/README.md``);
(c) every relative markdown link under ``wiki/`` resolves to an existing path
    (``#anchors`` and ``?queries`` are stripped; external URLs are skipped);
(d) catalogue-backed setup pages are current and their evidence shards verify.

Pure stdlib -- no PyAuto* imports, no PYTHONPATH.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADER_LABELS = (
    "Status",
    "Question",
    "Pre-registered rule",
    "Verdict",
    "Headline",
    "Library PRs",
    "Profiling PRs",
    "Ledger",
    "Mind contract",
    "Next",
)

# Inline markdown links / images: [text](target) or [text](target "title").
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_EXTERNAL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")  # http:, https:, mailto:, ...
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


def _strip_code(text: str) -> str:
    """Drop fenced code blocks and inline code spans, whose brackets are not links."""
    out, in_fence = [], False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def relative_links(md_path: Path) -> list[tuple[str, Path]]:
    """Return ``(raw_target, resolved_path)`` for every relative link in ``md_path``."""
    links = []
    for raw in _LINK_RE.findall(_strip_code(md_path.read_text(encoding="utf-8"))):
        if _EXTERNAL_RE.match(raw) or raw.startswith("#"):
            continue
        target = raw.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        links.append((raw, (md_path.parent / target).resolve()))
    return links


def check(root: Path) -> list[str]:
    """Return a list of human-readable failures (empty when the wiki is consistent)."""
    root = root.resolve()
    wiki = root / "wiki"
    notes = root / "results" / "notes"
    index = wiki / "index.md"
    campaigns = wiki / "campaigns"
    failures: list[str] = []

    if not wiki.is_dir():
        return [f"wiki/ directory missing under {root}"]
    if not index.is_file():
        failures.append("wiki/index.md missing")

    wiki_files = sorted(wiki.rglob("*.md"))
    linked: set[Path] = set()
    for md in wiki_files:
        for raw, resolved in relative_links(md):
            linked.add(resolved)
            # (c) every relative link resolves.
            if not resolved.exists():
                failures.append(f"(c) broken link in {md.relative_to(root)}: {raw}")

    # (a) every ledger is linked from the wiki.
    for note in sorted(notes.glob("*.md")) if notes.is_dir() else []:
        if note.resolve() not in linked:
            failures.append(f"(a) ledger not linked from wiki/: {note.relative_to(root)}")

    # (b) every campaign page is indexed and carries the full header.
    index_targets = (
        {resolved for _, resolved in relative_links(index)} if index.is_file() else set()
    )
    for page in sorted(campaigns.glob("*.md")) if campaigns.is_dir() else []:
        if page.name == "_template.md":
            continue
        rel = page.relative_to(root)
        if page.resolve() not in index_targets:
            failures.append(f"(b) campaign page has no row in wiki/index.md: {rel}")
        text = page.read_text(encoding="utf-8")
        missing = [lab for lab in HEADER_LABELS if f"**{lab}:**" not in text]
        if missing:
            failures.append(f"(b) {rel} missing header label(s): {', '.join(missing)}")

    if (root / "catalogue/registry.json").is_file():
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_setup_wiki import check as check_setups

        failures.extend(check_setups(root))

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 on any failure")
    parser.add_argument(
        "--root", type=Path, default=None, help="repo root (default: the ruff.toml dir)"
    )
    args = parser.parse_args(argv)
    root = args.root or _profiling_root()

    failures = check(root)
    if failures:
        print(f"check_wiki: {len(failures)} failure(s)")
        for f in failures:
            print(f"  {f}")
    else:
        n_pages = len(list((root / "wiki" / "campaigns").glob("*.md")))
        n_notes = len(list((root / "results" / "notes").glob("*.md")))
        print(f"check_wiki: OK ({n_pages} campaign files, {n_notes} ledgers linked)")
    return 1 if (failures and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
