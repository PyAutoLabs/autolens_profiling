"""Day-zero migration map: determinism, coverage, collision refusal and ``--check``."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1] / "tooling"
REPO = TOOLING.parents[2]
spec = importlib.util.spec_from_file_location(
    "build_migration_map", TOOLING / "build_migration_map.py"
)
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value if isinstance(value, str) else json.dumps(value))


def row(**extra):
    return {"autolens_version": "2026.10.4.1", "instrument": "hst", **extra}


@pytest.fixture
def tree(tmp_path):
    """A scratch git checkout with the real route manifest and a handful of results."""
    (tmp_path / "ruff.toml").write_text("")
    (tmp_path / "catalogue").mkdir()
    shutil.copy(REPO / mm.ROUTES_PATH, tmp_path / mm.ROUTES_PATH)
    write(tmp_path, "scripts/imaging/delaunay/likelihood_runtime.py", "")
    write(tmp_path, "scripts/misc/hazards/run.py", "")
    a100 = {"backend": "gpu", "hostname": "euclid-ral-gpu-2"}
    write(
        tmp_path,
        "results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64.json",
        row(device=a100, configuration={"inversion_path": "dense"}),
    )
    write(
        tmp_path,
        "results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64_r2026_10_17_1.json",
        row(autolens_version="2026.10.17.1", device=a100),
    )
    write(
        tmp_path,
        "results/runtime/imaging/delaunay/hst/delaunay_local_cpu_mp_sparse.json",
        row(device={"backend": "cpu"}),
    )
    write(tmp_path, "results/runtime/imaging/delaunay/hst/comparison.json", {"configs": {}})
    write(tmp_path, "results/breakdown/imaging/delaunay_hpc_a100_fp64_n3000.json", row())
    write(tmp_path, "results/breakdown/imaging/mystery_hpc_a100_fp64.json", row())
    write(
        tmp_path,
        "results/runtime/imaging/a100_probes/vmap_probe_delaunay.json",
        row(captured_at="2026-10-05T00:00:00Z"),
    )
    write(tmp_path, "results/hazards/hazards_index.json", {})
    write(tmp_path, "results/notes/design_lock_in.md", "# ledger\n")
    write(tmp_path, "results/notes/numpy_deflections_cpu.md", "# ledger\n")
    write(tmp_path, "wiki/campaigns/numpy_deflections_cpu.md", "# page\n")
    write(tmp_path, "results/README.md", "# results\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    return tmp_path


def by_old(doc):
    return {e["old"]: e for e in doc["entries"]}


def test_every_tracked_file_is_covered_exactly_once(tree):
    doc = mm.build(tree)
    olds = [e["old"] for e in doc["entries"]] + [u["old"] for u in doc["unresolved"]]
    assert sorted(olds) == mm.tracked(tree)
    assert len(olds) == len(set(olds))
    assert doc["counts"]["covered"] == doc["counts"]["tracked_files"] == len(olds)


def test_mapping_rules(tree):
    doc = mm.build(tree)
    entries = by_old(doc)
    base = "results/imaging/delaunay/hst"
    pinned = entries["results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64_r2026_10_17_1.json"]
    older = entries["results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64.json"]
    assert pinned["kind"] == "mirrored"
    assert pinned["new"] == f"{base}/jax_gpu_dense_fp64/hpc_a100/runtime.json"
    assert older["kind"] == "history"
    assert older["new"].startswith(f"{base}/jax_gpu_dense_fp64/hpc_a100/history/2026.10.4.1_")
    laptop = entries["results/runtime/imaging/delaunay/hst/delaunay_local_cpu_mp_sparse.json"]
    assert laptop["new"] == f"{base}/jax_cpu_sparse_mixed/local_laptop/runtime.json"
    assert entries["results/runtime/imaging/delaunay/hst/comparison.json"]["new"] == (
        f"{base}/comparison.json"
    )
    assert entries["results/hazards/hazards_index.json"]["new"] == (
        "results/misc/hazards/hazards_index.json"
    )
    probe = entries["results/runtime/imaging/a100_probes/vmap_probe_delaunay.json"]
    assert probe["kind"] == "archive" and probe["new"].startswith("archive/2026/10/probes/")
    assert entries["results/notes/design_lock_in.md"]["new"] == (
        "wiki/campaigns/misc/design_lock_in.md"
    )
    # A ledger that shares a campaign page's name is kept beside it, not merged.
    assert entries["results/notes/numpy_deflections_cpu.md"]["new"].endswith(
        "lens/deflections/numpy_deflections_cpu_ledger.md"
    )
    unresolved = {u["old"]: u["group"] for u in doc["unresolved"]}
    assert unresolved["results/breakdown/imaging/delaunay_hpc_a100_fp64_n3000.json"] == (
        "size-scan row"
    )
    assert unresolved["results/breakdown/imaging/mystery_hpc_a100_fp64.json"] == (
        "no canonical cell"
    )
    assert doc["mirror"]["result_variants"] == ["imaging/delaunay"]
    assert doc["mirror"]["result_variants_without_scripts"] == []


def test_build_is_deterministic(tree):
    first = mm.render(mm.build(tree))
    assert mm.render(mm.build(tree)) == first


def test_check_detects_a_stale_map(tree, capsys):
    assert mm.main(["--root", str(tree)]) == 0
    assert mm.main(["--root", str(tree), "--check"]) == 0
    write(tree, "results/notes/design_lock_in.md", "# ledger, edited\n")
    assert mm.main(["--root", str(tree), "--check"]) == 1
    assert "stale" in capsys.readouterr().err


def test_collisions_refuse_to_write(tree, monkeypatch, capsys):
    write(tree, "results/quick_update/hazards_index.json", {})
    _git(tree, "add", "-A")
    monkeypatch.setitem(mm.MISC_GROUPS, "quick_update", "hazards")
    doc = mm.build(tree)
    assert doc["collisions"] == [
        {
            "new": "results/misc/hazards/hazards_index.json",
            "old": [
                "results/hazards/hazards_index.json",
                "results/quick_update/hazards_index.json",
            ],
        }
    ]
    assert mm.main(["--root", str(tree)]) == 1
    assert not (tree / mm.MAP_PATH).exists()
    assert "COLLISION" in capsys.readouterr().err


def test_repository_map_has_no_collisions_and_covers_the_tree():
    doc = mm.build(REPO)
    assert doc["collisions"] == []
    assert doc["counts"]["covered"] == doc["counts"]["tracked_files"]
    news = [e["new"] for e in doc["entries"]]
    assert len(news) == len(set(news))
