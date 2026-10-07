"""Setup wiki transport, exact applicability and deterministic generation contracts."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "build_setup_wiki", ROOT / "scripts/misc/tooling/build_setup_wiki.py"
)
wiki = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wiki)


@pytest.fixture
def tree(tmp_path):
    index = json.loads((ROOT / "dashboard/catalogue.json").read_text())
    refs = index["evidence_shards"][:2]
    ids = {r["setup_id"] for r in refs}
    index["setups"] = [s for s in index["setups"] if s["id"] in ids] + [
        next(
            s
            for s in index["setups"]
            if s["id"] not in {r["setup_id"] for r in index["evidence_shards"]}
        )
    ]
    index["evidence_shards"] = refs
    index["records"] = [r for r in index["records"] if r["setup_id"] in ids]
    index["selections"] = [s for s in index["selections"] if s["setup_id"] in ids]
    index["recommendations"] = []
    for ref in refs:
        path = tmp_path / "dashboard" / ref["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / "dashboard" / ref["path"]).read_bytes())
    (tmp_path / "dashboard/catalogue.json").write_text(json.dumps(index))
    (tmp_path / "catalogue").mkdir()
    (tmp_path / "catalogue/wiki_bindings.json").write_text(json.dumps({"campaigns": []}))
    return tmp_path


def test_deterministic_generation_unknowns_absent_evidence_and_drift(tree):
    first = wiki.render_outputs(tree)
    assert first == wiki.render_outputs(tree)
    assert any("No measured evidence" in text for text in first.values())
    assert any("no current default substituted" in text for text in first.values())
    assert any("JSON pointer:" in text and "measurement/" in text for text in first.values())
    assert wiki.main(["--root", str(tree)]) == 0
    assert wiki.check(tree) == []
    page = tree / next(iter(first))
    page.write_text(wiki.MARKER + "\ndrift")
    assert any("stale generated" in f for f in wiki.check(tree))


def test_checksum_missing_shard_and_wrong_identity_fail(tree):
    index, _ = wiki.load_catalogue(tree)
    ref = index["evidence_shards"][0]
    path = tree / "dashboard" / ref["path"]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    assert "checksum mismatch" in wiki.check(tree)[0]
    doc = json.loads(original)
    doc["setups"][0]["id"] = "different"
    raw = json.dumps(doc).encode()
    path.write_bytes(raw)
    ref["sha256"] = hashlib.sha256(raw).hexdigest()
    (tree / "dashboard/catalogue.json").write_text(json.dumps(index))
    assert "identity/count mismatch" in wiki.check(tree)[0]
    path.unlink()
    assert "invalid setup wiki" in wiki.check(tree)[0]


def test_discovery_appears_on_model_index_without_exact_applicability(tree):
    index, shards = wiki.load_catalogue(tree)
    setup = index["setups"][0]
    index["unbound_findings"] = [
        {
            "id": "fixture-risk",
            "title": "Related fixture risk",
            "evidence": {"path": "results/hazards/hazards_index.json", "fragment": "/findings/f"},
            "discovery": {
                "models": [{"dataset": setup["dataset"], "model": setup["model"]}],
                "shared": False,
                "reason": "Fixture association only",
            },
        },
        {
            "id": "shared-risk",
            "title": "Shared component risk",
            "evidence": {"path": "results/hazards/hazards_index.json", "fragment": "/findings/g"},
            "discovery": {"models": [], "shared": True, "reason": "Not a model binding"},
        },
    ]
    outputs = wiki.render_outputs(tree, index, shards)
    model_page = f"wiki/setups/{setup['dataset']}/{setup['model']}/index.md"
    assert "Related fixture risk" in outputs[model_page]
    assert "Related fixture risk" not in outputs[wiki.page_path(setup)]
    assert "Shared component risk" in outputs["wiki/setups/findings.md"]
    assert "Related fixture risk" not in outputs["wiki/setups/findings.md"]


def test_recommendation_only_on_bound_exact_setup_and_no_campaign_inference(tree):
    index, shards = wiki.load_catalogue(tree)
    first, second = index["setups"][:2]
    record = shards[first["id"]]["records"][0]
    recommendation = copy.deepcopy(
        json.loads((ROOT / "dashboard/catalogue.json").read_text())["recommendations"][0]
    )
    recommendation["applies_to"]["setup_ids"] = [first["id"]]
    recommendation["record_ids"] = [record["id"]]
    recommendation["title"] = "Exact only candidate"
    recommendation["evidence"] = [first["evidence"]]
    index["recommendations"] = [recommendation]
    outputs = wiki.render_outputs(tree, index, shards)
    assert "Exact only candidate" in outputs[wiki.page_path(first)]
    assert "Exact only candidate" not in outputs[wiki.page_path(second)]
    assert "No campaign explicitly bound" in outputs[wiki.page_path(first)]


def test_invalid_campaign_binding_and_obsolete_cleanup_preserve_manual_pages(tree):
    (tree / "catalogue/wiki_bindings.json").write_text(
        json.dumps({"campaigns": [{"path": "wiki/campaigns/missing.md", "setup_ids": ["missing"]}]})
    )
    assert "campaign binding" in wiki.check(tree)[0]
    (tree / "catalogue/wiki_bindings.json").write_text(json.dumps({"campaigns": []}))
    stale = tree / "wiki/setups" / ("f" * 20 + ".md")
    stale.parent.mkdir(parents=True)
    stale.write_text(wiki.MARKER + "\nold")
    manual = stale.parent / "manual.md"
    manual.write_text("human journal")
    assert wiki.obsolete_pages(tree, {}) == [stale]
    wiki.main(["--root", str(tree)])
    assert not stale.exists()
    assert manual.read_text() == "human journal"


def test_live_links_and_support_remain_unreviewed():
    index, shards = wiki.load_catalogue(ROOT)
    for recommendation in index["recommendations"]:
        assert recommendation["validation"]["status"] == "unreviewed"
        assert recommendation["applies_to"]["constraints"]["measured_software_by_record"]
        for rid in recommendation["record_ids"]:
            records = [
                r
                for sid in recommendation["applies_to"]["setup_ids"]
                for r in shards[sid]["records"]
            ]
            record = next(r for r in records if r["id"] == rid)
            assert record["validation"]["status"] == "unreviewed"
            assert (
                record["identity"]["library_version"]
                in recommendation["applies_to"]["library_versions"]
            )


@pytest.mark.parametrize("model", ["../../../campaigns", "../..", "/absolute", "nested/model"])
def test_navigation_rejects_unsafe_components(tree, model):
    index, shards = wiki.load_catalogue(tree)
    index["setups"][-1]["model"] = model
    with pytest.raises(ValueError, match="Unsafe setup navigation"):
        wiki.render_outputs(tree, index, shards)


def test_generation_rejects_symlink_escape_and_unmarked_output(tree):
    destination = tree / "wiki/setups"
    destination.mkdir(parents=True)
    manual = destination / "index.md"
    manual.write_text("human-authored")
    with pytest.raises(ValueError, match="non-generated"):
        wiki.render_outputs(tree)
    manual.unlink()
    destination.rmdir()
    elsewhere = tree / "campaigns"
    elsewhere.mkdir()
    destination.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        wiki.render_outputs(tree)


def test_historical_matrix_uses_correct_mge_arm_and_exact_alma_analogues():
    index, shards = wiki.load_catalogue(ROOT)
    recommendations = {r["id"].rsplit("/", 1)[-1]: r for r in index["recommendations"]}
    setups = {s["id"]: s for s in index["setups"]}
    mge = recommendations["mge-sparse"]
    support = {
        r["id"]: r
        for sid in mge["applies_to"]["setup_ids"]
        for r in shards[sid]["records"]
        if r["id"] in mge["record_ids"]
    }
    assert "measurement/0107b28bf36bac6cdfc4" not in support
    sparse = support["measurement/3b9e72ef5d8791a738a9"]
    assert (
        sparse["evidence"]["fragment"]
        == "/jit_phases/library_sparse_full_pipeline/steady_per_call_s"
    )
    assert sparse["measurement"][sparse["metric"]] == pytest.approx(0.0037863500998355447)
    assert (
        support["measurement/d399e1ba20dcdf9665e0"]["evidence"]["fragment"]
        == "/full_pipeline_single_jit"
    )
    extent = recommendations["extent"]
    alma = {
        setups[sid]["evidence"]["path"]
        for sid in extent["applies_to"]["setup_ids"]
        if setups[sid]["instrument"] == "alma"
    }
    assert alma == {
        "results/breakdown/interferometer/" + model + "_" + suffix + ".json"
        for model in ["delaunay", "pixelization"]
        for suffix in ["hpc_a100_fp64", "numba_hpc_ral_cpu_fp64"]
    }
    assert len(extent["applies_to"]["setup_ids"]) == 8
    for sid in extent["applies_to"]["setup_ids"]:
        path = setups[sid]["evidence"]["path"]
        assert not any(excluded in path for excluded in ["n_sweep", "levers", "constant_split"])
        assert "_numba_" in path or "_a100_" in path
    for sid in recommendations["mesh-sparse"]["applies_to"]["setup_ids"]:
        path = setups[sid]["evidence"]["path"]
        assert {"path": path, "fragment": "/dense_total_step_by_step"} in recommendations[
            "mesh-sparse"
        ]["evidence"]
    for sid in recommendations["cpu-gate"]["applies_to"]["setup_ids"]:
        path = setups[sid]["evidence"]["path"]
        for pointer in [
            "/arms/numpy_fft/full_call/mean_s",
            "/arms/numba/full_call/mean_s",
            "/configuration/numba_gate_library_default",
            "/configuration/nnz_per_source_column",
        ]:
            assert {"path": path, "fragment": pointer} in recommendations["cpu-gate"]["evidence"]
