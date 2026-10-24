"""Producer evidence semantics, provenance boundaries and lazy transport."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[2] / "tooling"
spec = importlib.util.spec_from_file_location("build_catalogue", TOOLING / "build_catalogue.py")
cat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cat)
STAMP = "2026-10-05T00:00:00Z"
#: A mirrored cell (results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>/).
CELL = "results/imaging/delaunay/hst/jax_cpu_dense_fp64/hpc_ral_cpu"
EXAMPLE = f"{CELL}/runtime.json"


def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))


@pytest.fixture
def tree(tmp_path):
    reg = json.loads((TOOLING.parents[2] / "catalogue/registry.json").read_text())
    reg["references"] = []
    reg["recommendations"] = []
    reg["hazard_discovery"] = []
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


def date_all(root, day="2026-09-01"):
    """Record a first-commit date for every result file (what --refresh-commit-dates writes)."""
    dates = {
        p.relative_to(root).as_posix(): day for p in sorted((root / "results").rglob("*.json"))
    }
    write(root, cat.COMMIT_DATES, {"schema": "measured-at-first-commit", "dates": dates})


def update_registry(root, **values):
    reg = cat.registry(root)
    reg.update(values)
    write(root, cat.REGISTRY, reg)


def test_values_have_exact_original_anchors_and_unknowns(tree):
    path = EXAMPLE
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
        assert record["identity"]["precision"] == "float64"  # the setup folder says fp64
    setup = next(s for s in doc["setups"] if s["role"] != "planned_baseline")
    assert setup["configuration"]["source_pixels"]["value"] == 1200
    assert setup["configuration"]["psf_shape"]["value"] == [21, 21]
    assert setup["configuration"]["oversampling"]["value"] is None
    assert setup["configuration"]["oversampling"]["reason"]


def test_reference_is_explicit_not_latest_or_fastest(tree):
    path = EXAMPLE
    write(tree, path, row())
    write(
        tree,
        f"{CELL}/experiments/new/runtime.json",
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
    write(tree, EXAMPLE, row(full_pipeline_per_call=bad))
    doc = build(tree)
    assert not doc["records"]
    assert doc["inventory"][0]["status"] == "indexed_only"
    assert doc["inventory"][0]["reasons"]


def test_unversioned_result_is_indexed_without_inventing_stack(tree):
    write(tree, EXAMPLE, {"full_pipeline_per_call": 0.1})
    doc = build(tree)
    assert not doc["records"]
    assert "no measured software" in doc["inventory"][0]["reasons"][0]


def test_failed_streaming_is_not_a_successful_timing(tree):
    write(
        tree,
        "results/misc/streaming_scaling/run.json",
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
        f"{CELL}/experiments/vmap_probe/runtime.json",
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
    path = "results/misc/hazards/hazards_index.json"
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


def test_discovery_is_explicit_and_does_not_bind_applicability(tree):
    path = "results/misc/hazards/hazards_index.json"
    write(tree, path, {"findings": {"key": {"finding_id": "f1", "title": "Finding"}}})
    model = cat.registry(tree)["cells"][0]
    discovery = {
        "finding_id": "f1",
        "models": [{"dataset": model["dataset"], "model": model["model"]}],
        "shared": False,
        "reason": "Related fixture; exact applicability unknown.",
    }
    update_registry(tree, hazard_discovery=[discovery])
    doc = build(tree)
    assert not doc["hazards"]
    finding = doc["unbound_findings"][0]
    assert finding["status"] == "applicability_unknown"
    assert finding["discovery"]["models"] == discovery["models"]
    assert finding["evidence"]["path"] == path
    update_registry(tree, hazard_discovery=[{**discovery, "models": [], "shared": True}])
    assert build(tree)["unbound_findings"][0]["discovery"]["shared"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"finding_id": "missing"},
        {"models": [{"dataset": "imaging", "model": "not-a-model"}]},
        {"models": []},
        {"shared": True},
        {"reason": ""},
    ],
)
def test_invalid_hazard_discovery_fails_closed(tree, change):
    write(tree, "results/misc/hazards/hazards_index.json", {"findings": [{"finding_id": "f1"}]})
    model = cat.registry(tree)["cells"][0]
    item = {
        "finding_id": "f1",
        "models": [{"dataset": model["dataset"], "model": model["model"]}],
        "shared": False,
        "reason": "Related fixture only",
        **change,
    }
    update_registry(tree, hazard_discovery=[item])
    with pytest.raises(ValueError, match="hazard discovery"):
        build(tree)


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
    write(tree, EXAMPLE, row())
    date_all(tree)
    first = cat.render_outputs(tree, STAMP, "abc")
    assert first == cat.render_outputs(tree, STAMP, "abc")
    index = json.loads(first[cat.SETUPS_INDEX])
    assert not index["records"]  # no declared reference
    manifest = index["evidence_shards"][0]
    assert manifest["axes"] == ["runtime"]
    assert manifest["devices"] == ["cpu"]
    assert manifest["axis_devices"] == {"runtime": ["cpu"]}
    assert manifest["precisions"] == ["float64"]
    assert hashlib.sha256(first[manifest["path"]].encode()).hexdigest() == manifest["sha256"]
    shard = json.loads(first[manifest["path"]])
    assert len(shard["records"]) == manifest["records"] == index["archive_record_count"] == 1
    assert shard["setups"][0]["id"] == manifest["setup_id"]


def test_check_detects_missing_changed_obsolete_and_preserves_manual_files(tree):
    write(tree, EXAMPLE, row())
    date_all(tree)
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
    path = EXAMPLE
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
    path = EXAMPLE
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
    date_all(tree)
    index = json.loads(cat.render_outputs(tree, STAMP, "abc")[cat.SETUPS_INDEX])
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
        EXAMPLE,
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
    path = EXAMPLE
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


# ---------------------------------------------------------------- profiling-summary@3 ---

DELAUNAY_HST = {
    "id": "imaging/delaunay/hst",
    "dataset": "imaging",
    "model": "delaunay",
    "instrument": "hst",
}
A100 = "results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100"
LAPTOP = "results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu"


FIRST_COMMIT = "2026-09-01"


def lifecycle(root, dated=True):
    """Build v3; ``dated`` records a first-commit date for every result file in the tree."""
    update_registry(root, cells=[DELAUNAY_HST])
    if dated:
        date_all(root, FIRST_COMMIT)
    return cat.build_lifecycle(root, STAMP, "0123456789abcdef0123456789abcdef01234567")


def by_path(doc):
    return {r["evidence"]["path"]: r for r in doc["records"] if r["measurement"] == "runtime"}


def pulse_classify():
    for base in TOOLING.parents[3], TOOLING.parents[3].parent / "organs":
        if (base / "PyAutoPulse/pulse/summary.py").is_file():
            sys.path.insert(0, str(base / "PyAutoPulse"))
            from pulse.summary import classify

            return classify
    pytest.skip("PyAutoPulse checkout not beside this repo")


def test_v3_records_carry_the_lifecycle_fields_and_pass_pulse(tree):
    write(tree, f"{A100}/runtime.json", row(steps={"solve": 0.1}))
    doc = lifecycle(tree)
    assert (doc["schema"], doc["version"]) == ("profiling-summary", 3)
    assert "setups" not in doc and "selections" not in doc
    record = by_path(doc)[f"{A100}/runtime.json"]
    assert record["likelihood"] == {"dataset_type": "imaging", "variant": "delaunay"}
    assert (record["instrument"], record["setup_id"], record["host_class"]) == (
        "hst",
        "jax_gpu_dense_fp64",
        "hpc_a100",
    )
    assert record["setup"] == {
        "backend": "jax",
        "device": "gpu",
        "operator": "dense",
        "precision": "fp64",
        "transformer": "na",
        "threads": None,
    }
    assert record["metric"] == "single_call" and record["values"]["steps.solve"] == 0.1
    assert record["model_variant"] == "default" and record["status"] == "current"
    assert record["measured_at"] == f"{FIRST_COMMIT}T00:00:00Z"  # its file's first commit
    assert any(
        n.startswith("measured_at_derived") and "first committed" in n
        for n in record["limitations"]
    )
    assert doc["coverage"]["expected"]["cells"] == len(doc["planned_cells"])
    assert pulse_classify()(json.loads(json.dumps(doc)), ("profiling-summary", 3)) == ("ok", [])


def test_exactly_one_current_per_key_and_the_rest_superseded(tree):
    write(tree, f"{A100}/runtime.json", row())
    write(tree, f"{A100}/experiments/arm/runtime.json", row(full_pipeline_per_call=0.01))
    write(tree, f"{A100}/scans/n1250/runtime.json", row(configuration={"delaunay_vertices": 1250}))
    records = by_path(lifecycle(tree))
    current = records[f"{A100}/runtime.json"]
    assert current["status"] == "current"
    for path in (f"{A100}/experiments/arm/runtime.json", f"{A100}/scans/n1250/runtime.json"):
        assert records[path]["status"] == "superseded"
        assert records[path]["superseded_by"] == current["id"]  # never the faster or standard row


def test_local_rows_are_never_current_r7(tree):
    write(tree, f"{LAPTOP}/runtime.json", row())
    doc = lifecycle(tree)
    assert not doc["records"]  # no reference-host row: nothing can supersede it honestly
    [excluded] = [e for e in doc["coverage"]["excluded"] if e["key"]["measurement"] == "runtime"]
    assert excluded["record_count"] == 1 and "R7" in excluded["reason"]
    write(tree, f"{A100}/runtime.json", row())
    records = by_path(lifecycle(tree))
    assert records[f"{LAPTOP}/runtime.json"]["status"] == "superseded"
    assert (
        records[f"{LAPTOP}/runtime.json"]["superseded_by"] == records[f"{A100}/runtime.json"]["id"]
    )


def test_provisional_optimal_is_secondary_without_a_decision_r3(tree):
    write(tree, f"{A100}/runtime.json", row())
    write(tree, f"{A100.replace('fp64', 'mixed')}/runtime.json", row())
    doc = lifecycle(tree)
    records = by_path(doc)
    optimal = records[f"{A100}/runtime.json"]
    assert optimal["class"] == "secondary" and optimal["decision_id"] is None
    assert "provisional_optimal: true" in optimal["limitations"]
    assert records[f"{A100.replace('fp64', 'mixed')}/runtime.json"]["class"] == "experimental"
    assert not any(r["class"] == "optimal" for r in doc["records"])
    planned = {(c["setup_id"], c["host_class"]): c["class"] for c in doc["planned_cells"]}
    assert planned[("jax_gpu_dense_fp64", "hpc_a100")] == "optimal"
    assert planned[("numba_cpu_sparse_fp64", "hpc_ral_cpu")] == "optimal"
    assert planned[("jax_gpu_sparse_fp64", "hpc_a100")] == "secondary"
    assert not any("mixed" in sid for sid, _host in planned)


def test_standard_size_wins_and_fallback_is_labelled(tree):
    write(tree, f"{A100}/runtime.json", row(configuration={"delaunay_vertices": 1500}))
    record = by_path(lifecycle(tree))[f"{A100}/runtime.json"]
    assert record["status"] == "current" and record["regime"] == "source_pixels_1500"
    assert any(n.startswith("size_fallback") and "1250" in n for n in record["limitations"])
    configs = {
        "configs": {
            "big": row(configuration={"delaunay_vertices": 1500}),
            "standard": row(configuration={"delaunay_vertices": 1250}),
        }
    }
    write(tree, f"{A100}/runtime.json", configs)
    doc = lifecycle(tree)
    rows_ = {r["evidence"]["fragment"]: r for r in doc["records"] if r["measurement"] == "runtime"}
    standard, big = rows_["/configs/standard"], rows_["/configs/big"]
    assert standard["status"] == "current" and standard["regime"] == "standard"
    assert not any(n.startswith("size_fallback") for n in standard["limitations"])
    assert big["status"] == "superseded" and big["superseded_by"] == standard["id"]
    assert cat.lc.STANDARD_SOURCE_PIXELS == {
        "delaunay": 1250,
        "delaunay_nn": 1250,
        "rectangular": 1225,
    }


def test_unplanned_cells_scans_and_archive_are_not_current(tree):
    write(tree, "results/imaging/delaunay/sdp81/jax_gpu_dense_fp64/hpc_a100/runtime.json", row())
    write(tree, f"{A100}/scans/n3000/runtime.json", row())
    write(tree, "archive/2026/08/old/delaunay_hpc_a100_fp64.json", row(steps={"a": 1.0}))
    doc = lifecycle(tree)
    assert not doc["records"]
    reasons = " ".join(e["reason"] for e in doc["coverage"]["excluded"])
    assert "not a planned baseline cell" in reasons and "scan/experiment" in reasons
    assert doc["archive_record_count"] == 2  # runtime + breakdown rows, counted only


def test_fit_figure_is_attached_with_its_digest(tree):
    write(tree, f"{A100}/runtime.json", row())
    (tree / A100 / "fit.png").write_bytes(b"png")
    [figure] = by_path(lifecycle(tree))[f"{A100}/runtime.json"]["figures"]
    assert figure == {
        "kind": "fit",
        "path": f"{A100}/fit.png",
        "sha256": hashlib.sha256(b"png").hexdigest(),
    }


def test_fit_figures_match_the_pulse_v3_shape(tree):
    """``figures[]`` is Pulse REFERENCE v3's ``{kind, path, sha256}`` with a safe repo path.

    A fixture cell with a runtime and a breakdown row beside one ``fit.png`` (what a profiled
    cell leaves after 1B), plus an experiment arm without one: both cell rows carry the same
    figure, the arm carries none, and the whole document still validates in Pulse.
    """
    write(tree, f"{A100}/runtime.json", row())
    write(tree, f"{A100}/breakdown.json", row(steps={"solve": 0.1}))
    write(tree, f"{A100}/experiments/arm/runtime.json", row(full_pipeline_per_call=0.01))
    (tree / A100 / "fit.png").write_bytes(b"\x89PNG fixture")
    doc = lifecycle(tree)
    expected = {
        "kind": "fit",
        "path": f"{A100}/fit.png",
        "sha256": hashlib.sha256(b"\x89PNG fixture").hexdigest(),
    }
    paths = {record["evidence"]["path"] for record in doc["records"]}
    assert {f"{A100}/runtime.json", f"{A100}/breakdown.json"} <= paths
    for record in doc["records"]:
        path = record["evidence"]["path"]
        if path.startswith(f"{A100}/experiments/"):
            assert record["figures"] == [], path
        else:
            assert record["figures"] == [expected], path
            assert set(record["figures"][0]) == {"kind", "path", "sha256"}
            assert cat.safe_path(record["figures"][0]["path"])
    assert pulse_classify()(json.loads(json.dumps(doc)), ("profiling-summary", 3)) == ("ok", [])


def test_lifecycle_values_are_checked_against_the_evidence(tree):
    write(tree, f"{A100}/runtime.json", row())
    doc = lifecycle(tree)
    changed = copy.deepcopy(doc)
    changed["records"][0]["values"][changed["records"][0]["metric"]] = 999
    with pytest.raises(ValueError, match="original evidence"):
        cat.validate_lifecycle(tree, changed)
    second = copy.deepcopy(doc["records"][0])
    second["id"] = "lifecycle/duplicate-current"
    doc["records"].append(second)
    with pytest.raises(ValueError, match="Second current"):
        cat.validate_lifecycle(tree, doc)


def test_undated_row_without_a_first_commit_date_fails_check(tree, capsys):
    write(tree, f"{A100}/runtime.json", row())
    write(tree, cat.COMMIT_DATES, {"schema": "measured-at-first-commit", "dates": {}})
    with pytest.raises(cat.UndatedRows, match="--refresh-commit-dates") as missing:
        lifecycle(tree, dated=False)
    assert missing.value.paths == [f"{A100}/runtime.json"]
    assert cat.main(["--root", str(tree), "--check"]) == 1
    assert "--refresh-commit-dates" in capsys.readouterr().out


def test_timestamped_rows_need_no_first_commit_date(tree):
    stamped = row(device={"backend": "gpu", "provenance": {"captured_at": "2026-10-02T03:04:05Z"}})
    write(tree, f"{A100}/runtime.json", stamped)
    record = by_path(lifecycle(tree, dated=False))[f"{A100}/runtime.json"]
    assert record["measured_at"] == "2026-10-02T03:04:05Z"
    assert not any(n.startswith("measured_at_derived") for n in record["limitations"])


# ---------------------------------------------------------------- parity witnesses ---


def parity(**extra):
    """A synthetic parity.json witness: fixture numbers, not a measurement."""
    return {
        "schema": "parity-witness",
        "version": 1,
        "status": "match",
        "log_likelihood": -100.0,
        "reference_log_likelihood": -100.004,
        "tolerance": 0.01,
        "reference": {"source": "fixture reference", "date": "2026-10-01"},
        "measured_at": "2026-10-02T03:04:05Z",
        "autolens_version": "2026.10.17.1",
        **extra,
    }


def test_parity_witness_becomes_a_v3_parity_record_and_passes_pulse(tree):
    write(tree, f"{A100}/runtime.json", row())
    write(tree, f"{A100}/parity.json", parity(source_pixels=1250))
    doc = lifecycle(tree)
    [record] = [r for r in doc["records"] if r["measurement"] == "parity"]
    assert record["status"] == "current" and record["regime"] == "standard"
    assert (record["metric"], record["unit"]) == ("abs_delta_log_likelihood", "dimensionless")
    assert record["values"]["abs_delta_log_likelihood"] == pytest.approx(0.004)
    assert record["values"]["tolerance"] == 0.01
    assert record["parity"]["status"] == "match"
    assert record["parity"]["reference_log_likelihood"] == -100.004
    assert record["release_pin"] == "2026.10.17.1"
    assert record["measured_at"] == "2026-10-02T03:04:05Z"
    [witness] = doc["parity_witnesses"]
    assert witness["record_id"] == record["id"] and witness["status"] == "match"
    assert pulse_classify()(json.loads(json.dumps(doc)), ("profiling-summary", 3)) == ("ok", [])
    # The companion v2 catalogue keeps the file indexed, never as a timing.
    [source] = [s for s in build(tree)["inventory"] if s["path"] == f"{A100}/parity.json"]
    assert source["record_count"] == 0 and "Parity witness" in source["reasons"][0]


def test_not_assessed_witness_is_listed_but_yields_no_record(tree):
    write(tree, f"{A100}/runtime.json", row())
    write(
        tree,
        f"{A100}/parity.json",
        parity(
            status="not_assessed",
            log_likelihood=None,
            reference_log_likelihood=None,
            tolerance=None,
            reason="reference run pending",
        ),
    )
    doc = lifecycle(tree)
    assert not [r for r in doc["records"] if r["measurement"] == "parity"]
    [witness] = doc["parity_witnesses"]
    assert witness["status"] == "not_assessed" and witness["record_id"] is None


def test_invalid_witness_fails_the_build(tree):
    write(tree, f"{A100}/parity.json", parity(status="diverged"))
    with pytest.raises(ValueError, match="Invalid parity witness"):
        lifecycle(tree)


def test_history_witness_is_superseded_and_tampering_is_refused(tree):
    write(tree, f"{A100}/parity.json", parity())
    write(tree, f"{A100}/history/2026.8.17.1_2026-09-01/parity.json", parity(tolerance=0.02))
    doc = lifecycle(tree)
    rows_ = {r["evidence"]["path"]: r for r in doc["records"] if r["measurement"] == "parity"}
    current = rows_[f"{A100}/parity.json"]
    old = rows_[f"{A100}/history/2026.8.17.1_2026-09-01/parity.json"]
    assert current["status"] == "current" and old["superseded_by"] == current["id"]
    changed = copy.deepcopy(doc)
    [rec] = [r for r in changed["records"] if r["id"] == current["id"]]
    rec["values"]["abs_delta_log_likelihood"] = 0.0
    with pytest.raises(ValueError, match="parity.json witness"):
        cat.validate_lifecycle(tree, changed)
