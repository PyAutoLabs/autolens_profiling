"""``_profile_cli``'s legacy-name helpers: where a file a writer names by its pre-#442 path is.

Writers that keep a legacy name (``results/runtime/...``, ``results/breakdown/...``,
``results/<old group>/...``) and the readers that parse those names agree through
``legacy_landing_dir`` / ``legacy_read_dirs`` / ``locate_legacy_result`` /
``legacy_result_files`` / ``legacy_result_dirs`` (autolens_profiling#442, PR 3b). The
committed-row cases read the real day-zero map; nothing here writes under ``results/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT))

import _profile_cli as pc  # noqa: E402

MAP = json.loads((ROOT / "catalogue/migration_day_zero.json").read_text())
MOVES = {e["old"]: e["new"] for e in MAP["entries"] if e["new"] != e["old"]}


@pytest.mark.parametrize(
    ("legacy", "landing"),
    [
        ("results/runtime/imaging/mge", "results/misc/likelihood_runtime/imaging/mge"),
        ("results/breakdown/imaging", "results/misc/likelihood_breakdown/imaging"),
        ("results/simulators", "results/misc/simulators"),
        ("results/sweep", "results/misc/likelihood_breakdown"),
        ("results/streaming_scaling/x", "results/misc/streaming_scaling/x"),
    ],
)
def test_the_landing_folder_of_each_legacy_folder(legacy, landing):
    assert pc.legacy_landing_dir(ROOT / legacy) == ROOT / landing
    assert pc.legacy_read_dirs(ROOT / legacy) == [ROOT / landing, ROOT / legacy]


def test_paths_outside_the_legacy_folders_pass_through(tmp_path):
    for path in (tmp_path / "out", ROOT / "results/imaging/delaunay", ROOT / "output/x"):
        assert pc.legacy_landing_dir(path) == path
        assert pc.legacy_read_dirs(path) == [path]
        assert pc.locate_legacy_result(path / "absent.json") is None


def _moved(prefix: str, suffix: str = ".json") -> tuple[str, str]:
    return next(
        (o, n)
        for o, n in sorted(MOVES.items())
        if o.startswith(prefix)
        and o.endswith(suffix)
        and n.startswith("results/")
        and not n.startswith("results/misc/")
        and (ROOT / n).is_file()
    )


def test_a_committed_row_is_found_by_its_legacy_name_after_the_move():
    old, new = _moved("results/breakdown/imaging/")
    assert pc.locate_legacy_result(ROOT / old) == ROOT / new
    files = dict((name, path) for path, name in pc.legacy_result_files(ROOT / Path(old).parent))
    assert files[Path(old).name] == ROOT / new


def test_a_fresh_row_is_found_where_the_placement_puts_it(tmp_path, monkeypatch):
    """A row the map never saw is looked up at ``destination(legacy, doc)``."""
    legacy = ROOT / "results/breakdown/imaging/delaunay_hpc_a100_fp64_n77777.json"
    placed = pc.mirrored_output_path(legacy, {"instrument": "hst"})
    assert "/scans/n77777/" in placed.as_posix()
    seen = []
    monkeypatch.setattr(Path, "exists", lambda self: seen.append(self) or self == placed)
    assert pc.locate_legacy_result(legacy, {"instrument": "hst"}) == placed
    assert seen[0] == legacy  # the legacy path itself (an unresolved row) is tried first


def test_a_slot_the_map_gave_another_legacy_name_is_not_borrowed():
    """``<cell>/runtime.json`` from a 2-level folder is not the 3-level folder's row."""
    old, new = _moved("results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64")
    other = ROOT / "results/runtime/imaging/delaunay/hst" / Path(old).name
    assert pc.locate_legacy_result(other, {"instrument": "hst"}) != ROOT / new


def test_result_files_keep_names_from_the_landing_and_legacy_folders(tmp_path):
    folder = tmp_path / "cell"
    folder.mkdir()
    (folder / "b.json").write_text("{}")
    (folder / "a.json").write_text("{}")
    (folder / "a.png").write_text("")
    assert pc.legacy_result_files(folder) == [
        (folder / "a.json", "a.json"),
        (folder / "b.json", "b.json"),
    ]


def test_result_dirs_list_the_legacy_folders_the_map_moved():
    dirs = pc.legacy_result_dirs(ROOT / "results/runtime")
    assert ROOT / "results/runtime/imaging/mge/hst" in dirs
    assert all(d.is_relative_to(ROOT / "results/runtime") for d in dirs)
