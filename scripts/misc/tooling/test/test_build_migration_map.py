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
    write(tmp_path, "scripts/interferometer/delaunay/likelihood_breakdown.py", "")
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
    write(tmp_path, "results/breakdown/imaging/delaunay_hpc_a100_fp64_recon_split.json", row())
    write(
        tmp_path,
        "results/breakdown/imaging/fixed_light_trace_delaunay_psf_control_hpc_a100_fp64"
        "_fixed_light_trace.json",
        row(),
    )
    write(
        tmp_path,
        "results/breakdown/imaging/delaunay_breakdown_hst_v2026.8.17.1.json",
        row(device={"backend": "cpu"}),
    )
    write(
        tmp_path,
        "results/breakdown/interferometer/sma/delaunay_hpc_ral_cpu_fp64.json",
        row(
            instrument="sma",
            transformer="TransformerNUFFT",
            configuration={"inversion_path": "sparse"},
            device={"backend": "cpu", "hostname": "euclid-ral-gpu-1"},
        ),
    )
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
    a100 = f"{base}/jax_gpu_dense_fp64/hpc_a100"
    assert entries["results/breakdown/imaging/delaunay_hpc_a100_fp64_n3000.json"]["new"] == (
        f"{a100}/scans/n3000/breakdown.json"
    )
    arm = entries["results/breakdown/imaging/delaunay_hpc_a100_fp64_recon_split.json"]
    assert arm["kind"] == "experiment"
    assert arm["new"] == f"{a100}/experiments/recon_split/breakdown.json"
    fixed = next(e for e in doc["entries"] if "fixed_light_trace" in e["old"])
    assert fixed["new"] == f"{a100}/light_fixed/experiments/trace_psf_control/breakdown.json"
    assert entries["results/breakdown/imaging/mystery_hpc_a100_fp64.json"]["new"] == (
        "results/misc/likelihood_breakdown/imaging/mystery_hpc_a100_fp64.json"
    )
    hostless = entries["results/breakdown/imaging/delaunay_breakdown_hst_v2026.8.17.1.json"]
    assert hostless["new"].startswith("archive/2026/08/hostless_summaries/")
    # CPU on a RAL GPU node is hpc_ral_cpu; a sparse interferometric row is the W~ route.
    wtilde = entries["results/breakdown/interferometer/sma/delaunay_hpc_ral_cpu_fp64.json"]
    assert wtilde["new"] == (
        "results/interferometer/delaunay/sma/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json"
    )
    assert doc["unresolved"] == []
    assert doc["mirror"]["result_variants"] == ["imaging/delaunay", "interferometer/delaunay"]
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
    committed = json.loads((REPO / mm.MAP_PATH).read_text())
    if mm.applied(REPO, committed):
        # After the move-only PR (#442 PR 2) the map is frozen: every move is in the tree.
        moved = [e for e in committed["entries"] if e["new"] != e["old"]]
        assert all((REPO / e["new"]).is_file() for e in moved)
        # Nothing is left at an old path except new pointer READMEs (different bytes).
        assert not any(
            (REPO / e["old"]).exists() and mm.sha256(REPO, e["old"]) == e["sha256"] for e in moved
        )
        assert all((REPO / u["old"]).is_file() for u in committed["unresolved"])
        return
    doc = mm.build(REPO)
    assert doc["collisions"] == []
    assert doc["counts"]["covered"] == doc["counts"]["tracked_files"]
    news = [e["new"] for e in doc["entries"]]
    assert len(news) == len(set(news))
