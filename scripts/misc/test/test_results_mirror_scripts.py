"""The mirrored results tree mirrors ``scripts/`` (autolens_profiling#442, phase 0B witness).

Every ``results/<dataset_type>/<variant>/`` has a ``scripts/<dataset_type>/<variant>/``.
A script variant with no results yet is reported, never a failure. ``archive/``,
``results/misc/`` and ``results/lens/`` (component axis) are outside the mirror; the legacy
top-level folders the day-zero move emptied hold only their pointer README and the map's
unresolved rows.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
MAP = json.loads((ROOT / "catalogue/migration_day_zero.json").read_text())
OUTSIDE = {"misc", "lens"}


def _legacy_tops() -> set[str]:
    return {e["old"].split("/")[1] for e in MAP["entries"] if e["old"].startswith("results/")} - (
        OUTSIDE | {"README.md", "provenance_grandfathered.txt"}
    )


def _dirs(base: Path) -> set[tuple[str, str]]:
    return {
        (d.parent.name, d.name)
        for d in base.glob("*/*")
        if d.is_dir() and d.parent.name not in OUTSIDE and d.name != "__pycache__"
    }


def _mirror() -> tuple[set, set]:
    legacy = _legacy_tops()
    results = {
        (dt, v)
        for dt, v in _dirs(ROOT / "results")
        if dt not in legacy or (ROOT / "scripts" / dt).is_dir()
    }
    scripts = _dirs(ROOT / "scripts")
    return results, scripts


def test_every_results_variant_has_a_scripts_variant():
    results, scripts = _mirror()
    assert results, "the mirrored tree is empty"
    assert results <= scripts, f"results variants with no scripts/: {sorted(results - scripts)}"


def test_scripts_without_results_are_reported_not_failed(capsys):
    results, scripts = _mirror()
    script_only = sorted(f"{dt}/{v}" for dt, v in scripts - results)
    with capsys.disabled():
        print(f"\nscript variants with no results yet (reported, not a failure): {script_only}")
    assert all(isinstance(v, str) for v in script_only)


def test_emptied_legacy_folders_hold_only_their_pointer_and_unresolved_rows():
    unresolved = {u["old"] for u in MAP["unresolved"]}
    for top in sorted(_legacy_tops()):
        if (ROOT / "scripts" / top).is_dir():
            continue  # a dataset type that is also a mirrored folder
        folder = ROOT / "results" / top
        if not folder.is_dir():
            continue
        leftovers = sorted(
            p.relative_to(ROOT).as_posix()
            for p in folder.rglob("*")
            if p.is_file() and p.name != "README.md"
        )
        assert set(leftovers) <= unresolved, f"results/{top}/ still holds {leftovers}"
        assert (folder / "README.md").is_file(), f"results/{top}/ lacks its pointer README"


def test_archive_is_outside_the_mirror():
    assert not (ROOT / "results" / "archive").exists()
    assert (ROOT / "archive" / "README.md").is_file()
