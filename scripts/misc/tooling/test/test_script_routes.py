"""Migration contracts: no compute on lookup/dry-run and stable legacy outputs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT))
import _script_routes as routes


def load_sweep():
    spec = importlib.util.spec_from_file_location(
        "migration_sweep", ROOT / "scripts/misc/likelihood_runtime/sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_sources_compile_and_retired_aliases_resolve_without_execution():
    manifest = routes.load_routes()
    for row in manifest["routes"]:
        path = ROOT / row["path"]
        compile(path.read_bytes(), str(path), "exec")
        assert not (ROOT / row["legacy"]).exists()
        assert not list((ROOT / row["legacy"]).parent.rglob("*.py"))
        assert routes.canonical_path(row["legacy"]) == path
        assert routes.canonical_path(path) == path
        assert routes.legacy_stem(path) == Path(row["legacy"]).stem
    assert {row["dataset"] for row in manifest["routes"]} == {
        "imaging",
        "interferometer",
        "datacube",
        "cluster",
        "multi_dataset",
        "point_source_image",
        "point_source_source",
    }


def test_published_matrix_preserves_legacy_sweep_cells():
    sweep = load_sweep()
    cells = routes.load_routes()["runtime_cells"]
    assert [(r["dataset"], r["model"], tuple(r["instruments"])) for r in cells] == sweep.CELLS
    for row in cells:
        assert routes.runtime_path(row["dataset"], row["model"]) == ROOT / row["path"]
    assert routes.runtime_path("imaging", "rectangular") == routes.runtime_path(
        "imaging", "pixelization"
    )


def test_dry_sweep_does_not_spawn_or_write_results(monkeypatch, tmp_path):
    sweep = load_sweep()

    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run attempted execution")

    monkeypatch.setattr(sweep.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["sweep", "--dry-run", "--output-root", str(tmp_path)])
    assert sweep.main() == 0
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.log"))


@pytest.mark.parametrize("model", ["pixelization", "delaunay_nn", "mge"])
def test_timeout_marker_keeps_historical_model_name(monkeypatch, tmp_path, model):
    sweep = load_sweep()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(sweep.subprocess, "run", timeout)
    ok, _, _ = sweep._run_one(
        "python",
        routes.runtime_path("imaging", model),
        sweep.CONFIGS[0],
        tmp_path,
        False,
        sparse=True,
        per_run_timeout=1,
    )
    assert ok  # Existing CPU-unusable classification is a successful terminal result.
    marker = tmp_path / f"{model}_local_cpu_fp64_sparse.unusable.json"
    assert json.loads(marker.read_text())["cpu_unusable"] is True
    assert not list(tmp_path.glob("likelihood_runtime_*.json"))


def test_alias_lookup_does_not_require_legacy_file_but_requires_canonical(tmp_path):
    legacy = "scripts/imaging/likelihood_runtime/mge.py"
    canonical = "scripts/imaging/mge/likelihood_runtime.py"
    target = tmp_path / canonical
    target.parent.mkdir(parents=True)
    target.write_text("raise RuntimeError('lookup must not execute')")
    (tmp_path / "catalogue").mkdir()
    (tmp_path / "catalogue/script_routes.json").write_text(
        json.dumps(
            {
                "schema": "profiling-script-routes",
                "version": 1,
                "routes": [{"legacy": legacy, "path": canonical}],
            }
        )
    )
    assert routes.canonical_path(legacy, tmp_path) == target
    assert routes.legacy_stem(target, tmp_path) == "mge"
    target.unlink()
    with pytest.raises(ValueError, match="Missing script route"):
        routes.load_routes(tmp_path)


def test_unknown_route_fails_without_guessing():
    with pytest.raises(ValueError, match="Unknown"):
        routes.runtime_path("imaging", "not_a_model")


@pytest.mark.parametrize("bad", ["../escape.py", "/tmp/x.py", "scripts/missing.py"])
def test_bad_manifest_route_fails_closed(tmp_path, bad):
    (tmp_path / "catalogue").mkdir()
    (tmp_path / "catalogue/script_routes.json").write_text(
        json.dumps(
            {
                "schema": "profiling-script-routes",
                "version": 1,
                "routes": [{"path": bad, "legacy": bad}],
            }
        )
    )
    with pytest.raises(ValueError):
        routes.load_routes(tmp_path)
