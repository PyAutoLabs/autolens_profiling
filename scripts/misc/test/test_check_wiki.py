"""Unit tests for ``tooling/check_wiki.py`` — the profiling research wiki drift gate.

Each test builds a miniature repo (``wiki/`` + ``results/notes/``) under
``tmp_path`` and runs the checker against it. Pure stdlib; no JAX dependency.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_check_wiki.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_spec = importlib.util.spec_from_file_location(
    "check_wiki", _profiling_root() / "scripts" / "misc" / "tooling" / "check_wiki.py"
)
check_wiki = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_wiki)

HEADER = """**Status:** open
**Question:** q
**Pre-registered rule:** r
**Verdict:** v
**Headline:** h
**Library PRs:** none
**Profiling PRs:** none
**Ledger:** [ledger](../../results/notes/alpha.md)
**Mind contract:** none
**Next:** none
"""


def _build(root: Path, header: str = HEADER, link_beta: bool = True) -> Path:
    (root / "results" / "notes").mkdir(parents=True)
    (root / "wiki" / "campaigns").mkdir(parents=True)
    (root / "results" / "notes" / "alpha.md").write_text("# alpha\n")
    (root / "results" / "notes" / "beta.md").write_text("# beta\n")
    (root / "wiki" / "campaigns" / "_template.md").write_text("# template\n")
    (root / "wiki" / "campaigns" / "camp.md").write_text(f"# Camp\n\n{header}\n## Journal\n")
    beta = "[beta](../results/notes/beta.md)" if link_beta else "beta"
    (root / "wiki" / "index.md").write_text(
        f"| Campaign | Ledger |\n|---|---|\n| [Camp](campaigns/camp.md) | {beta} |\n"
    )
    return root


def test__consistent_tree_passes(tmp_path):
    root = _build(tmp_path)
    assert check_wiki.check(root) == []
    assert check_wiki.main(["--check", "--root", str(root)]) == 0


def test__unlinked_ledger_fails(tmp_path, capsys):
    root = _build(tmp_path, link_beta=False)
    failures = check_wiki.check(root)
    assert failures == ["(a) ledger not linked from wiki/: results/notes/beta.md"]
    assert check_wiki.main(["--check", "--root", str(root)]) == 1
    assert "results/notes/beta.md" in capsys.readouterr().out


def test__missing_header_label_fails(tmp_path):
    root = _build(tmp_path, header=HEADER.replace("**Verdict:** v\n", ""))
    failures = check_wiki.check(root)
    assert len(failures) == 1
    assert "wiki/campaigns/camp.md missing header label(s): Verdict" in failures[0]


def test__broken_link_and_unindexed_page_fail(tmp_path):
    root = _build(tmp_path)
    (root / "wiki" / "campaigns" / "orphan.md").write_text(
        f"# Orphan\n\n{HEADER}\n[gone](../../results/notes/gone.md#anchor)\n"
    )
    failures = check_wiki.check(root)
    assert any("(b) campaign page has no row" in f and "orphan.md" in f for f in failures)
    assert any("(c) broken link" in f and "gone.md#anchor" in f for f in failures)


def test__external_links_and_code_spans_are_ignored(tmp_path):
    root = _build(tmp_path)
    page = root / "wiki" / "campaigns" / "camp.md"
    page.write_text(
        page.read_text()
        + "[pr](https://github.com/PyAutoLabs/PyAutoArray/pull/1)\n`[x](nowhere.md)`\n"
    )
    assert check_wiki.check(root) == []
