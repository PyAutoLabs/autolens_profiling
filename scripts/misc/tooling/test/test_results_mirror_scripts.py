"""The mirrored results tree mirrors ``scripts/`` (autolens_profiling#442, phase 0B witness).

Every ``results/<dataset_type>/<variant>/`` has a ``scripts/<dataset_type>/<variant>/`` and
vice versa; every ``results/lens/<component>/`` has its ``scripts/lens/<component>/`` and vice
versa; every ``results/misc/<group>/`` names a ``scripts/misc/<group>/`` or a documented
non-script group. ``archive/`` is outside the mirror. The legacy top-level folders the day-zero
move emptied hold only their pointer README and the map's unresolved rows.

Principled exclusions (each is asserted, so the lists cannot go stale silently):

- ``SHARED_SCRIPT_VARIANTS``: script folders shared by several mesh variants; their results
  land in the mesh's own cell (``imaging/pixelized/fixed_light*.py`` writes
  ``results/imaging/{delaunay,rectangular}/...``).
- ``MODEL_VARIANT_SCRIPTS``: a script variant whose results are a ``<model_variant>`` folder of
  another likelihood variant.
- ``NOT_YET_MEASURED``: script variants with no committed result yet. Committing the first
  result must remove the entry.
- ``MISC_WITHOUT_SCRIPTS``: ``results/misc/`` groups no single ``scripts/misc/`` group writes.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
MAP = json.loads((ROOT / "catalogue/migration_day_zero.json").read_text())
OUTSIDE = {"misc", "lens"}

SHARED_SCRIPT_VARIANTS = {
    ("imaging", "pixelized"): ("delaunay", "rectangular"),
    ("interferometer", "pixelized"): ("delaunay", "rectangular"),
}
MODEL_VARIANT_SCRIPTS = {("imaging", "mge_mass"): ("rectangular", "mge_mass")}
NOT_YET_MEASURED = {
    ("imaging", "sersic"),
    ("multi_dataset", "delaunay"),
}
MISC_WITHOUT_SCRIPTS = {
    "logs": "committed SLURM job logs, one folder per campaign; provenance, never a result",
    "streaming_scaling": "written by scripts/interferometer/delaunay/ streaming cells",
}


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


def _children(base: Path) -> set[str]:
    return {d.name for d in base.iterdir() if d.is_dir() and d.name != "__pycache__"}


def test_every_results_variant_has_a_scripts_variant():
    results, scripts = _mirror()
    assert results, "the mirrored tree is empty"
    assert results <= scripts, f"results variants with no scripts/: {sorted(results - scripts)}"


def test_every_scripts_variant_has_results_or_a_documented_exclusion():
    results, scripts = _mirror()
    excluded = set(SHARED_SCRIPT_VARIANTS) | set(MODEL_VARIANT_SCRIPTS) | NOT_YET_MEASURED
    missing = sorted(f"{dt}/{v}" for dt, v in scripts - results - excluded)
    assert not missing, f"scripts variants with no results/ and no documented exclusion: {missing}"


def test_the_exclusions_are_still_true():
    results, scripts = _mirror()
    for (dt, v), meshes in SHARED_SCRIPT_VARIANTS.items():
        assert (dt, v) in scripts and (dt, v) not in results
        assert all((dt, mesh) in results for mesh in meshes), (dt, v)
    for (dt, v), (variant, model_variant) in MODEL_VARIANT_SCRIPTS.items():
        assert (dt, v) in scripts and (dt, v) not in results
        cells = ROOT.glob(f"results/{dt}/{variant}/*/*/*/{model_variant}")
        assert any(p.is_dir() for p in cells), f"{dt}/{v}: no {variant} cell has {model_variant}/"
    for dt, v in NOT_YET_MEASURED:
        assert (dt, v) in scripts, f"{dt}/{v} is no longer a script variant"
        assert (dt, v) not in results, f"{dt}/{v} now has results: drop it from NOT_YET_MEASURED"


def test_lens_components_mirror_both_ways():
    results = _children(ROOT / "results/lens")
    scripts = _children(ROOT / "scripts/lens")
    assert results == scripts, (sorted(results - scripts), sorted(scripts - results))


def test_misc_groups_name_a_script_group_or_a_documented_exception():
    results = _children(ROOT / "results/misc")
    scripts = _children(ROOT / "scripts/misc")
    assert set(MISC_WITHOUT_SCRIPTS) <= results
    unexplained = sorted(results - scripts - set(MISC_WITHOUT_SCRIPTS))
    assert not unexplained, f"results/misc groups with no scripts/misc group: {unexplained}"


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
