"""Producer evidence semantics, provenance boundaries and lazy transport."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1] / "tooling"
spec = importlib.util.spec_from_file_location("build_catalogue", TOOLING / "build_catalogue.py")
cat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cat)
STAMP = "2026-10-05T00:00:00Z"


def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))


@pytest.fixture
def tree(tmp_path):
    reg = json.loads((TOOLING.parents[2] / "catalogue/registry.json").read_text())
    reg["references"] = []
    reg["recommendations"] = []
    reg["cells"] = [reg["cells"][0]]
    write(tmp_path, cat.REGISTRY, reg)
    write(tmp_path, "dashboard/summary.json", {"generated_at": STAMP, "producer_revision": "abc"})
    return tmp_path


def row(**extra):
    return {
        "autolens_version": "2026.10.1",
        "instrument": "hst",
        "model": "delaunay",
        "device": {"backend": "cpu"},
        "full_pipeline_per_call": 0.3,
        "configuration": {"delaunay_vertices": 1200, "psf_shape": [21, 21]},
        **extra,
    }


def build(root):
    return cat.build(root, STAMP, "abc")


def update_registry(root, **values):
    reg = cat.registry(root)
    reg.update(values)
    write(root, cat.REGISTRY, reg)


def test_values_have_exact_original_anchors_and_unknowns(tree):
    path = "results/runtime/imaging/delaunay/example.json"
    write(tree, path, row(steps={"mapper/a~b": 0.12}, jit_phases={"solve": {"compile_s": 2.0}}))
    doc = build(tree)
    assert {r["axis"] for r in doc["records"]} == {"runtime", "breakdown", "compile"}
    for record in doc["records"]:
        assert (
            cat.resolve_pointer(
                json.loads((tree / path).read_text()), record["evidence"]["fragment"]
            )
            == record["measurement"][record["metric"]]
        )
        assert record["validation"]["status"] == "unreviewed"
        assert record["provenance"]["qualified"] is False
        assert record["method"]["synchronization"] is None
        assert record["identity"]["precision"] is None
    setup = next(s for s in doc["setups"] if s["role"] != "planned_baseline")
    assert setup["configuration"]["source_pixels"]["value"] == 1200
    assert setup["configuration"]["psf_shape"]["value"] == [21, 21]
    assert setup["configuration"]["oversampling"]["value"] is None
    assert setup["configuration"]["oversampling"]["reason"]


def test_reference_is_explicit_not_latest_or_fastest(tree):
    path = "results/runtime/imaging/delaunay/old.json"
    write(tree, path, row())
    write(
        tree,
        "results/runtime/imaging/delaunay/new.json",
        row(autolens_version="2099.1.1", full_pipeline_per_call=0.001),
    )
    update_registry(
        tree,
        references=[
            {
                "path": path,
                "pointer": "",
                "reason": "Explicit illustrative candidate, not accepted baseline.",
            }
        ],
    )
    doc = build(tree)
    selected = [r for r in doc["records"] if r["id"] in {s["record_id"] for s in doc["selections"]}]
    assert [r["evidence"]["path"] for r in selected] == [path]
    assert all(s["status"] == "unreviewed" for s in doc["selections"])
    assert len({r["setup_id"] for r in doc["records"]}) == 2
    assert all(
        c["status"] == "not_measured" and c["record_id"] is None for c in doc["planned_cells"]
    )
    assert "not-a-measurement" not in json.dumps(doc)


def test_missing_reference_fails_closed(tree):
    update_registry(
        tree,
        references=[
            {"path": "results/missing.json", "pointer": "", "reason": "Declared reference"}
        ],
    )
    with pytest.raises(ValueError, match="did not resolve"):
        build(tree)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, None])
def test_invalid_values_never_become_measurements(tree, bad):
    write(tree, "results/runtime/imaging/bad.json", row(full_pipeline_per_call=bad))
    doc = build(tree)
    assert not doc["records"]
    assert doc["inventory"][0]["status"] == "indexed_only"
    assert doc["inventory"][0]["reasons"]


def test_unversioned_result_is_indexed_without_inventing_stack(tree):
    write(tree, "results/runtime/imaging/unversioned.json", {"full_pipeline_per_call": 0.1})
    doc = build(tree)
    assert not doc["records"]
    assert "no measured software" in doc["inventory"][0]["reasons"][0]


def test_failed_streaming_is_not_a_successful_timing(tree):
    write(
        tree,
        "results/streaming_scaling/run.json",
        {
            "autolens_version": "2026.10.1",
            "hardware": "local_cpu",
            "rows": [{"outcome": "MEMORY_LIMIT", "wall_s": 8, "peak_rss_mb": 100}],
            "arms": {
                "bounded": {
                    "rows": [
                        {
                            "outcome": "OK",
                            "wall_s": 3,
                            "peak_rss_mb": 50,
                            "n_vis": 10000,
                            "chunk": 1000,
                        }
                    ]
                }
            },
        },
    )
    doc = build(tree)
    assert len(doc["records"]) == 2
    assert all("/arms/bounded/rows/0/" in r["evidence"]["fragment"] for r in doc["records"])
    assert {s["status"] for s in doc["selections"]} == {"unusable"}
    assert all(r["axis"] != "runtime" for r in doc["records"])


def test_static_memory_stays_estimate_and_never_becomes_recommendation(tree):
    write(
        tree,
        "results/runtime/imaging/vmap_probe_delaunay.json",
        row(samples=[{"peak_bytes": 123}], per_replica_mb=12, recommended_batch_size=64),
    )
    doc = build(tree)
    assert len(doc["static_memory_estimates"]) == 1
    assert not any(r["axis"] == "memory" for r in doc["records"])
    assert not doc["recommendations"]


def test_compile_probe_is_not_full_likelihood_and_redacts_local_paths(tree):
    write(
        tree,
        "scripts/misc/jax_compile/results/probe.json",
        [
            {
                "jax_version": "0.7.2",
                "compile_s": 10,
                "steady_s": 0.1,
                "configuration": {
                    "cache_dir": "/mnt/ral/private",
                    "note": "/home/person/cache",
                    "centre": [-1.0, 2.0],
                },
                "timestamp": "2026-10-01T12:00:00",
                "transform": "vmap",
                "batch_size": 4,
            }
        ],
    )
    doc = build(tree)
    assert {r["axis"] for r in doc["records"]} == {"compile", "breakdown"}
    assert all(r["provenance"]["measured_at"] is None for r in doc["records"])
    assert "/mnt/" not in json.dumps(doc) and "/home/" not in json.dumps(doc)
    setup = next(s for s in doc["setups"] if s["role"] != "planned_baseline")
    assert setup["configuration"]["centre"]["value"] == [-1.0, 2.0]
    assert setup["configuration"]["batch_size"]["value"] == 4


def test_hazard_remains_unbound_and_has_exact_finding_anchor(tree):
    path = "results/hazards/hazards_index.json"
    write(
        tree,
        path,
        {
            "findings": {
                "key/a": {"finding_id": "f1", "title": "Hazard", "summary": "Observed risk"}
            }
        },
    )
    doc = build(tree)
    assert not doc["hazards"]
    finding = doc["unbound_findings"][0]
    assert finding["evidence"]["fragment"] == "/findings/key~1a"
    assert (
        cat.resolve_pointer(json.loads((tree / path).read_text()), finding["evidence"]["fragment"])[
            "finding_id"
        ]
        == "f1"
    )


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/bad", "a//b", "a/./b", "a\\b"])
def test_paths_cannot_escape_checkout(tree, path):
    with pytest.raises(ValueError, match="Unsafe"):
        cat.source_path(tree, path)


def test_symlink_escape_refused(tree):
    outside = tree.parent / (tree.name + "-outside.json")
    outside.write_text("{}")
    (tree / "escape.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        cat.source_path(tree, "escape.json")


def test_shards_are_deterministic_complete_and_content_addressed(tree):
    write(tree, "results/runtime/imaging/delaunay/example.json", row())
    first = cat.render_outputs(tree, STAMP, "abc")
    assert first == cat.render_outputs(tree, STAMP, "abc")
    index = json.loads(first["catalogue.json"])
    assert not index["records"]  # no declared reference
    manifest = index["evidence_shards"][0]
    assert manifest["axes"] == ["runtime"]
    assert manifest["devices"] == ["cpu"]
    assert manifest["precisions"] == ["precision not recorded"]
    assert hashlib.sha256(first[manifest["path"]].encode()).hexdigest() == manifest["sha256"]
    shard = json.loads(first[manifest["path"]])
    assert len(shard["records"]) == manifest["records"] == index["archive_record_count"] == 1
    assert shard["setups"][0]["id"] == manifest["setup_id"]


def test_check_detects_missing_changed_obsolete_and_preserves_manual_files(tree):
    write(tree, "results/runtime/imaging/delaunay/example.json", row())
    assert cat.main(["--root", str(tree), "--check"]) == 1
    assert cat.main(["--root", str(tree)]) == 0
    assert cat.main(["--root", str(tree), "--check"]) == 0
    stale = tree / "dashboard/catalogue/shards" / ("f" * 20 + ".json")
    stale.write_text("{}")
    manual = stale.with_name("manual.json")
    manual.write_text("{}")
    assert cat.main(["--root", str(tree), "--check"]) == 1
    assert cat.main(["--root", str(tree)]) == 0
    assert not stale.exists() and manual.exists()
    (tree / "dashboard/catalogue.json").write_text("{}")
    assert cat.main(["--root", str(tree), "--check"]) == 1


def test_mutated_measurement_and_selection_mismatch_refused(tree):
    path = "results/runtime/imaging/delaunay/example.json"
    write(tree, path, row())
    update_registry(tree, references=[{"path": path, "pointer": "", "reason": "Example"}])
    doc = build(tree)
    changed = copy.deepcopy(doc)
    changed["records"][0]["measurement"]["single_call"] = 999
    with pytest.raises(ValueError, match="original evidence"):
        cat.validate_local(tree, changed)
    doc["selections"][0]["method_id"] = "other-method"
    with pytest.raises(ValueError, match="identity mismatch"):
        cat.validate_local(tree, doc)


def test_recommendation_cannot_accept_unreviewed_evidence(tree):
    path = "results/runtime/imaging/delaunay/example.json"
    write(tree, path, row())
    record = build(tree)["records"][0]
    rec = {
        "id": "recommendation/example",
        "title": "Test advice",
        "description": "Do not accept automatically",
        "applies_to": {
            "setup_ids": [record["setup_id"]],
            "library_versions": ["2026.10.1"],
            "limitations": "This exact test setup only",
            "constraints": {},
        },
        "evidence": [{"path": path, "fragment": None}],
        "record_ids": [record["id"]],
        "validation": {"status": "accepted", "reason": "Unsupported acceptance"},
    }
    update_registry(tree, recommendations=[rec])
    with pytest.raises(ValueError, match="unaccepted"):
        build(tree)
    rec["validation"]["status"] = "unreviewed"
    update_registry(tree, recommendations=[rec])
    index = json.loads(cat.render_outputs(tree, STAMP, "abc")["catalogue.json"])
    assert index["records"][0]["id"] == record["id"]


def test_script_navigation_never_imports_scientific_code(tree):
    path = tree / "scripts/imaging/likelihood_runtime/delaunay.py"
    path.parent.mkdir(parents=True)
    path.write_text("raise RuntimeError('must never import')")
    assert build(tree)["navigation"] == [
        {
            "path": path.relative_to(tree).as_posix(),
            "dataset": "imaging",
            "model": "delaunay",
            "category": "scientific_entrypoint",
        }
    ]


def test_runtime_compile_phases_and_batched_breakdown_remain_distinct(tree):
    write(
        tree,
        "results/runtime/imaging/delaunay/example.json",
        row(
            full_pipeline_lower_s=1.0,
            full_pipeline_compile_s=3.0,
            full_pipeline_first_call_s=0.7,
            steps_vmap_per_call={"solve": 0.02},
            vmap_batch=16,
            psf_shape=[31, 31],
        ),
    )
    doc = build(tree)
    compiled = {
        r["metric"]: r["measurement"][r["metric"]] for r in doc["records"] if r["axis"] == "compile"
    }
    assert compiled == {
        "full_pipeline.lower": 1.0,
        "full_pipeline.compile": 3.0,
        "full_pipeline.first_call": 0.7,
    }
    assert any(r["metric"] == "steps_vmap_per_call.solve" for r in doc["records"])
    setup = next(s for s in doc["setups"] if s["role"] != "planned_baseline")
    assert setup["configuration"]["vmap_batch"]["value"] == 16
    assert setup["configuration"]["psf_shape"]["value"] == [31, 31]


def test_recommendation_rejects_invalid_exact_evidence_pointer(tree):
    path = "results/runtime/imaging/delaunay/example.json"
    write(tree, path, row())
    doc = build(tree)
    record = doc["records"][0]
    doc["recommendations"] = [
        {
            "id": "test/pointer",
            "title": "Historical candidate",
            "description": "Exact support required",
            "applies_to": {
                "setup_ids": [record["setup_id"]],
                "library_versions": [record["identity"]["library_version"]],
                "constraints": {},
                "limitations": "Unreviewed historical fixture",
            },
            "record_ids": [record["id"]],
            "evidence": [{"path": path, "fragment": "/not-recorded"}],
            "validation": {"status": "unreviewed", "reason": "fixture"},
        }
    ]
    with pytest.raises(ValueError, match="Invalid finding evidence pointer"):
        cat.validate_local(tree, doc)
    doc["recommendations"][0]["evidence"][0] = record["evidence"]
    cat.validate_local(tree, doc)
