"""The day-zero move tooling: applier, reference rewrite, legacy reader view, writer placement.

``apply_migration_map.py`` moves the tree and repoints references from
``catalogue/migration_day_zero.json`` alone; ``results_layout.py`` gives readers the
pre-move identity of a moved file; ``_profile_cli.resolve_output_paths`` places a new run in
the mirrored tree with the map generator's own classification (autolens_profiling#442 PR 2).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
TOOLING = ROOT / "scripts" / "misc" / "tooling"
sys.path.insert(0, str(TOOLING))
sys.path.insert(0, str(ROOT))

import results_layout as rl  # noqa: E402

_spec = importlib.util.spec_from_file_location("apply_mm", TOOLING / "apply_migration_map.py")
amm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(amm)

OLD = "results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64.json"
NEW = "results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/runtime.json"
OLD_NOTE = "results/notes/alpha.md"
NEW_NOTE = "wiki/campaigns/imaging/delaunay/alpha.md"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _repo(tmp_path: Path) -> Path:
    """A tiny pre-move checkout: one result, one ledger linking it, one wiki page, a map."""
    root = tmp_path / "repo"
    (root / "results/runtime/imaging/delaunay").mkdir(parents=True)
    (root / OLD).write_text('{"full_pipeline_per_call": 0.1}\n')
    (root / "results/notes").mkdir(parents=True)
    (root / OLD_NOTE).write_text(
        f"# alpha\n\nRow: [json](../runtime/imaging/delaunay/{Path(OLD).name}).\n"
    )
    (root / "wiki").mkdir()
    (root / "wiki/index.md").write_text(f"[alpha](../{OLD_NOTE}) and `{OLD}`.\n")
    (root / "catalogue").mkdir()
    entries = [
        {"old": OLD, "new": NEW, "sha256": amm.sha256(root / OLD), "kind": "mirrored"},
        {
            "old": OLD_NOTE,
            "new": NEW_NOTE,
            "sha256": amm.sha256(root / OLD_NOTE),
            "kind": "wiki_campaign",
        },
    ]
    (root / amm.MAP_PATH).write_text(json.dumps({"entries": entries, "unresolved": []}))
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    return root


def test_apply_moves_with_git_mv_and_is_rerunnable(tmp_path):
    root = _repo(tmp_path)
    doc = json.loads((root / amm.MAP_PATH).read_text())
    pending, done, errors = amm.plan(root, doc)
    assert (len(pending), done, errors) == (2, 0, [])
    for old, new in pending:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        _git(root, "mv", old, new)
    assert (root / NEW).is_file() and not (root / OLD).exists()
    pending, done, errors = amm.plan(root, doc)
    assert (pending, done, errors) == ([], 2, [])


def test_apply_refuses_a_changed_file(tmp_path):
    root = _repo(tmp_path)
    (root / OLD).write_text("{}\n")
    _pending, _done, errors = amm.plan(root, json.loads((root / amm.MAP_PATH).read_text()))
    assert any("sha256 differs" in e for e in errors)


def test_rewrite_refs_relinks_moved_pages_and_literal_paths_idempotently(tmp_path):
    root = _repo(tmp_path)
    doc = json.loads((root / amm.MAP_PATH).read_text())
    for old, new in amm.plan(root, doc)[0]:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        _git(root, "mv", old, new)
    # results/** is never rewritten: the moved ledger is under wiki/ now, so it is.
    changed = amm.rewrite_refs(root, doc)
    assert sorted(changed) == [NEW_NOTE, "wiki/index.md"]
    index = (root / "wiki/index.md").read_text()
    assert "[alpha](campaigns/imaging/delaunay/alpha.md)" in index
    assert f"`{NEW}`" in index
    ledger = (root / NEW_NOTE).read_text()
    assert "[json](../../../../results/imaging/delaunay/hst/" in ledger
    assert amm.rewrite_refs(root, doc) == []


def test_legacy_view_reads_moved_files_under_their_old_names(tmp_path):
    root = _repo(tmp_path)
    doc = json.loads((root / amm.MAP_PATH).read_text())
    for old, new in amm.plan(root, doc)[0]:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        _git(root, "mv", old, new)
    (root / "results/runtime/imaging/delaunay/fixture_local_cpu_fp64.json").write_text("{}")
    rl._maps.cache_clear()
    found = rl.iter_legacy(root, "results/runtime", "*.json")
    assert [(p.relative_to(root).as_posix(), str(legacy)) for p, legacy in found] == [
        (NEW, OLD),
        (
            "results/runtime/imaging/delaunay/fixture_local_cpu_fp64.json",
            "results/runtime/imaging/delaunay/fixture_local_cpu_fp64.json",
        ),
    ]
    assert rl.legacy_of(root, NEW) == OLD and rl.current_of(root, OLD) == NEW
    assert rl.current(root / OLD) == root / NEW
    assert rl.legacy_glob(root / "results/runtime/imaging", "*.json") == []
    assert [p.name for p in rl.legacy_glob(root / "results/runtime/imaging/delaunay")] == [
        "runtime.json",
        "fixture_local_cpu_fp64.json",
    ]


def test_destination_uses_the_map_classification():
    assert rl.destination(ROOT, OLD, {"instrument": "hst"}) == (NEW, "mirrored")
    sparse, kind = rl.destination(
        ROOT, "results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64_sparse.json", {}
    )
    assert kind == "misc"  # no instrument recorded: never a guessed cell
    assert sparse.startswith("results/misc/likelihood_runtime/")
    probe, kind = rl.destination(
        ROOT, "results/breakdown/imaging/delaunay_hpc_a100_fp64_probe.json", {"instrument": "hst"}
    )
    assert kind == "misc" and probe.startswith("results/misc/")  # never archived on write


def test_resolve_output_paths_places_default_dirs_and_honours_explicit_ones(tmp_path):
    import _profile_cli as pc

    cli = pc.parse_profile_cli(argv=["--config-name", "hpc_a100_fp64"])
    doc = pc._setup_hints(cli, "delaunay_likelihood_summary_hst_v1", None)
    assert doc["instrument"] == "hst" and "device" not in doc
    placed = pc.mirrored_output_path(ROOT / OLD, doc)
    assert placed == ROOT / NEW
    outside = tmp_path / "out" / "delaunay_hpc_a100_fp64.json"
    assert pc.mirrored_output_path(outside, doc) == outside
    explicit = pc.parse_profile_cli(
        argv=["--config-name", "hpc_a100_fp64", "--output-dir", str(tmp_path / "x")]
    )
    json_path, png_path = pc.resolve_output_paths(
        explicit, ROOT / "results/runtime/imaging/delaunay", "delaunay_likelihood_summary_hst_v1"
    )
    assert json_path == (tmp_path / "x" / "delaunay_hpc_a100_fp64.json").resolve()
    assert png_path.suffix == ".png"


def test_place_output_path_skips_paths_outside_the_legacy_folders(tmp_path):
    import _profile_cli as pc

    cli = SimpleNamespace(config_name=None, instrument=None, use_mixed_precision=False)
    path = tmp_path / "results/breakdown/imaging/x.json"
    assert pc.place_output_path(path, cli, "x") == path
