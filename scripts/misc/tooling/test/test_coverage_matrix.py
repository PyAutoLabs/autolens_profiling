"""Coverage matrix: cell states, the size-fallback ruling, the stale rule, parity witnesses,
and the Pulse designation mirror + holds as the one source catalogue_lifecycle.py reads."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1]
REPO = TOOLING.parents[2]
spec = importlib.util.spec_from_file_location("coverage_matrix", TOOLING / "coverage_matrix.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
sys.path.insert(0, str(TOOLING))
import catalogue_lifecycle as lc  # noqa: E402

DESIGNATIONS = json.loads((REPO / cm.DESIGNATIONS).read_text())
HOLDS = json.loads((REPO / cm.HOLDS).read_text())
LIKELIHOOD = {"dataset_type": "imaging", "variant": "delaunay"}


def cell(setup_id="jax_gpu_dense_fp64", host="hpc_a100", measurement="runtime", cls="optimal"):
    return {
        "likelihood": LIKELIHOOD,
        "instrument": "hst",
        "setup_id": setup_id,
        "host_class": host,
        "model_variant": "default",
        "measurement": measurement,
        "class": cls,
    }


def record(rid, status="current", pin="2026.10.17.1", limitations=(), **key):
    row = cell(**key)
    row.pop("class")
    return {
        **row,
        "id": rid,
        "class": "secondary",
        "status": status,
        "release_pin": pin,
        "regime": "standard",
        "limitations": list(limitations),
    }


def catalogue(records, planned=None, witnesses=()):
    return {
        "schema": "profiling-summary",
        "version": 3,
        "generated_at": "2026-10-24T00:00:00Z",
        "records": records,
        "planned_cells": planned if planned is not None else [cell()],
        "parity_witnesses": list(witnesses),
    }


def states(cov):
    return {c["cell"]: c["state"] for c in cov["cells"]}


def only(cov):
    [c] = cov["cells"]
    return c


def test_current_superseded_and_missing():
    cov = cm.build(catalogue([record("a"), record("b", pin="2026.10.17.1")]), DESIGNATIONS, HOLDS)
    assert only(cov)["state"] == "current" and only(cov)["record"] == "a"
    cov = cm.build(catalogue([record("old", status="superseded")]), DESIGNATIONS, HOLDS)
    assert only(cov)["state"] == "superseded"
    cov = cm.build(catalogue([]), DESIGNATIONS, HOLDS)
    assert only(cov)["state"] == "missing" and only(cov)["reason"] == "no record"


def test_size_fallback_current_counts_as_missing():
    note = "size_fallback: no row at the standard 1250 source pixels exists for this key"
    cov = cm.build(catalogue([record("a", limitations=[note])]), DESIGNATIONS, HOLDS)
    assert only(cov)["state"] == "missing"
    assert only(cov)["reason"].startswith("size_fallback")
    assert cov["totals"]["missing_size_fallback"] == 1


def test_stale_is_a_pin_behind_the_catalogue_pin():
    # The catalogue pin is the newest record pin when the envelope declares none.
    newer = record("n", pin="2026.10.17.1", measurement="breakdown")
    planned = [cell(), cell(measurement="breakdown")]
    cov = cm.build(catalogue([record("a", pin="2026.8.17.1"), newer], planned), DESIGNATIONS, HOLDS)
    assert cov["catalogue_release_pin"] == "2026.10.17.1"
    assert cov["release_pin_source"] == "newest record"
    by = {c["measurement"]: c for c in cov["cells"]}
    assert by["runtime"]["state"] == "stale" and by["breakdown"]["state"] == "current"
    # Numeric, not lexical: 2026.10 is newer than 2026.9.
    cov = cm.build(catalogue([record("a", pin="2026.9.30.1"), newer], planned), DESIGNATIONS, HOLDS)
    assert {c["state"] for c in cov["cells"]} == {"stale", "current"}
    # An envelope pin wins; a record without a pin is stale.
    doc = catalogue([record("a", pin=None)])
    doc["release_pin"] = "2026.10.1.1"
    cov = cm.build(doc, DESIGNATIONS, HOLDS)
    assert cov["release_pin_source"] == "envelope" and only(cov)["state"] == "stale"


def test_local_current_never_covers_a_cell():
    doc = catalogue([record("a", host="local_gpu")], planned=[cell(host="local_gpu")])
    assert only(cm.build(doc, DESIGNATIONS, HOLDS))["state"] == "superseded"


def test_matrix_groups_likelihood_class_measurement():
    planned = [cell(), cell(setup_id="jax_gpu_sparse_fp64", cls="secondary")]
    cov = cm.build(catalogue([record("a")], planned), DESIGNATIONS, HOLDS)
    rows = {(r["likelihood"], r["class"], r["measurement"]): r for r in cov["matrix"]}
    assert rows[("imaging/delaunay", "optimal", "runtime")]["current"] == 1
    assert rows[("imaging/delaunay", "secondary", "runtime")]["missing"] == 1
    assert cov["totals"]["planned_cells"] == 2


def test_class_mismatch_against_the_designations_copy_is_reported():
    planned = [cell(setup_id="numba_cpu_sparse_fp64", host="hpc_ral_cpu", cls="secondary")]
    cov = cm.build(catalogue([], planned), DESIGNATIONS, HOLDS)
    [mismatch] = cov["class_mismatches"]
    assert mismatch["planned"] == "secondary" and mismatch["designated"] == "optimal"


def test_parity_states_per_setup_cell():
    planned = [cell(), cell(measurement="breakdown")]
    parity = record("p", measurement="parity")
    parity["parity"] = {"status": "diverged"}
    cov = cm.build(catalogue([parity], planned), DESIGNATIONS, HOLDS)
    assert cov["parity"]["totals"] == {
        "setup_cells": 1,
        "match": 0,
        "diverged": 1,
        "not_assessed": 0,
        "missing": 0,
    }
    witness = {**cell(), "status": "not_assessed", "location": "cell", "path": "x"}
    cov = cm.build(catalogue([], planned, [witness]), DESIGNATIONS, HOLDS)
    assert cov["parity"]["cells"][0]["state"] == "not_assessed"
    history = {**witness, "location": "history"}
    cov = cm.build(catalogue([], planned, [history]), DESIGNATIONS, HOLDS)
    assert cov["parity"]["cells"][0]["state"] == "missing"


def test_parity_records_are_not_unplanned():
    parity = record("p", measurement="parity")
    parity["parity"] = {"status": "match"}
    stray = record("s", setup_id="jax_cpu_na_fp64", host="local_laptop")
    cov = cm.build(catalogue([record("a"), parity, stray]), DESIGNATIONS, HOLDS)
    assert cov["unplanned_current_records"] == ["s"]
    assert cov["totals"]["current_records_unplanned"] == 1
    assert cov["parity"]["cells"][0]["state"] == "match"


def test_check_detects_drift(tmp_path, capsys):
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "catalogue").mkdir()
    (tmp_path / cm.CATALOGUE).write_text(json.dumps(catalogue([record("a")])))
    (tmp_path / cm.DESIGNATIONS).write_text(json.dumps(DESIGNATIONS))
    (tmp_path / cm.HOLDS).write_text(json.dumps(HOLDS))
    assert cm.main(["--root", str(tmp_path), "--check"]) == 1
    assert cm.main(["--root", str(tmp_path)]) == 0
    assert cm.main(["--root", str(tmp_path), "--check"]) == 0
    assert "| 1 | 1 | 0 | 0 | 0 (0) | 0 |" in (tmp_path / cm.OUT_MD).read_text()
    (tmp_path / cm.CATALOGUE).write_text(json.dumps(catalogue([])))
    assert cm.main(["--root", str(tmp_path), "--check"]) == 1
    assert "STALE" in capsys.readouterr().out
    bad = catalogue(
        [], [cell(setup_id="numba_cpu_sparse_fp64", host="hpc_ral_cpu", cls="secondary")]
    )
    (tmp_path / cm.CATALOGUE).write_text(json.dumps(bad))
    assert cm.main(["--root", str(tmp_path), "--check"]) == 1


def _designated_setups():
    rows = DESIGNATIONS["designations"]
    return (
        {r["optimal"] for r in rows}
        | {sid for r in rows for sid in r["secondary"]}
        | {sid for h in HOLDS["holds"] for sid in h["optimal"] + h["secondary"]}
    )


def test_the_mirror_holds_the_day_zero_designations():
    rows = DESIGNATIONS["designations"]
    assert len(rows) == 39
    assert {r["hardware_class"] for r in rows} == {"cpu_many_core", "gpu_a100"}
    assert {r["regime"] for r in rows} == {"standard"}
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    # The one held group is not designated (human 2026-10-31: does not separate at sma).
    [hold] = HOLDS["holds"]
    assert (hold["likelihood"], hold["hardware_class"]) == ("datacube/delaunay", "gpu_a100")
    assert ("datacube/delaunay", "gpu_a100") not in {
        (r["likelihood"], r["hardware_class"]) for r in rows
    }
    assert HOLDS["footnote"].startswith("Mixed precision is `experimental`")
    assert not (REPO / "catalogue" / "provisional_designations.json").exists()


def test_designations_json_is_the_single_source():
    # catalogue_lifecycle holds no copy of a designation: it reads the mirror coverage_matrix reads.
    assert lc.DESIGNATIONS_PATH == (REPO / cm.DESIGNATIONS).resolve()
    assert lc.HOLDS_PATH == (REPO / cm.HOLDS).resolve()
    assert {k: v["id"] for k, v in lc._DESIGNATIONS.items()} == {
        (r["likelihood"], r["hardware_class"]): r["id"] for r in DESIGNATIONS["designations"]
    }
    assert lc.HARDWARE_CLASSES == cm.HARDWARE_CLASSES
    literals = {
        node.value
        for node in ast.walk(ast.parse((TOOLING / "catalogue_lifecycle.py").read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not literals & _designated_setups(), "a designated setup_id is hard-coded"


def test_editing_the_mirror_changes_catalogue_lifecycle(tmp_path):
    doc = json.loads((REPO / cm.DESIGNATIONS).read_text())
    [row] = [
        r
        for r in doc["designations"]
        if (r["likelihood"], r["hardware_class"]) == ("imaging/delaunay", "cpu_many_core")
    ]
    row.update(optimal="jax_cpu_sparse_fp64", secondary=[])
    (tmp_path / "catalogue").mkdir()
    (tmp_path / cm.DESIGNATIONS).write_text(json.dumps(doc))
    (tmp_path / cm.HOLDS).write_text(json.dumps(HOLDS))
    copy = tmp_path / "scripts" / "misc" / "tooling" / "catalogue_lifecycle.py"
    copy.parent.mkdir(parents=True)
    copy.write_text((TOOLING / "catalogue_lifecycle.py").read_text())
    spec = importlib.util.spec_from_file_location("catalogue_lifecycle_edited", copy)
    edited = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edited)
    args = ("imaging", "delaunay", "hpc_ral_cpu")
    assert lc.planned_setups(*args) == {
        "jax_cpu_sparse_fp64": "secondary",
        "numba_cpu_sparse_fp64": "optimal",
    }
    assert edited.planned_setups(*args) == {"jax_cpu_sparse_fp64": "optimal"}
    assert edited.designation(*args, "numba_cpu_sparse_fp64") is None


def test_record_class_follows_the_designations():
    decision = "imaging-delaunay-standard-cpu-many-core"
    ral = ("imaging", "delaunay", "hpc_ral_cpu")
    assert lc.record_class(*ral, "numba_cpu_sparse_fp64") == ("optimal", decision, [])
    assert lc.record_class(*ral, "jax_cpu_sparse_fp64") == ("secondary", decision, [])
    # A setup measured on the key but not named by its designation (ruling 2026-10-31).
    assert lc.record_class(*ral, "jax_cpu_dense_fp64") == ("experimental", None, [])
    assert lc.record_class(*ral, "jax_cpu_sparse_mixed") == ("experimental", None, [])
    # Off the reference host or the standard regime: the role is inherited, never optimal.
    for args in (
        ("imaging", "delaunay", "local_laptop", "numba_cpu_sparse_fp64"),
        (*ral, "numba_cpu_sparse_fp64", "source_pixels_1500"),
    ):
        cls, decision_id, [note] = lc.record_class(*args)
        assert (cls, decision_id) == ("secondary", None)
        assert note.startswith(lc.UNDESIGNATED) and "optimal role" in note and decision in note
    # The held group keeps its provisional label, with no decision.
    held = ("datacube", "delaunay", "hpc_a100")
    assert lc.record_class(*held, "jax_gpu_sparse_fp64_wtilde") == (
        "secondary",
        None,
        ["provisional_optimal: true"],
    )
    assert lc.record_class(*held, "jax_gpu_dense_fp64_nufft") == (
        "secondary",
        None,
        ["provisional_secondary: true"],
    )
    assert lc.planned_setups(*held, "sma")["jax_gpu_sparse_fp64_wtilde"] == "optimal"


def test_the_catalogue_cites_only_the_mirrored_designations():
    doc = json.loads((REPO / cm.CATALOGUE).read_text())
    rows = {r["id"]: r for r in DESIGNATIONS["designations"]}
    records = {r["id"]: r for r in doc["records"]}
    for record in doc["records"]:
        if record["class"] == "optimal":
            assert record["decision_id"] in rows, record["id"]
        if record["decision_id"] is not None:
            row = rows[record["decision_id"]]
            assert record["setup_id"] in [row["optimal"], *row["secondary"]]
            assert (
                record["regime"] == "standard"
                and record["host_class"] in lc.REFERENCE_HOSTS.values()
            )
    # Every evidence id a designation cites is a record, declared with that decision.
    for row in DESIGNATIONS["designations"]:
        for rid in row["evidence"]:
            assert rid in records, (row["id"], rid)
    for row in DESIGNATIONS["designations"]:
        named = [r for r in records.values() if r["decision_id"] == row["id"]]
        assert any(r["class"] == "optimal" and r["status"] == "current" for r in named), row["id"]
    held = [
        r
        for r in doc["records"]
        if (r["likelihood"]["dataset_type"], r["likelihood"]["variant"], r["host_class"])
        == ("datacube", "delaunay", "hpc_a100")
    ]
    assert held and all(r["class"] != "optimal" and r["decision_id"] is None for r in held)
    assert any("provisional_optimal: true" in r["limitations"] for r in held)


def test_a_declared_size_variant_row_is_never_coverage():
    """A current size-variant row (#468; ``size_variant`` limitation) sits on its own catalogue
    key beside the standard row; coverage reads the standard row only, in either order."""
    note = "size_variant: a declared size measured beside the standard 1225 source pixels"
    sized = record("sized", limitations=[note])
    for records in ([sized, record("std")], [record("std"), sized]):
        cov = cm.build(catalogue(records), DESIGNATIONS, HOLDS)
        assert (only(cov)["state"], only(cov)["record"]) == ("current", "std")
        assert cov["unplanned_current_records"] == []
    cov = cm.build(catalogue([sized]), DESIGNATIONS, HOLDS)
    assert (only(cov)["state"], only(cov)["record"]) == ("missing", None)
    witness = {**cell(measurement="parity"), "status": "match", "location": "cell"}
    witness.pop("class")
    witnesses = [{**witness, "size_variant": "source_pixels_784"}]
    cov = cm.build(catalogue([], witnesses=witnesses), DESIGNATIONS, HOLDS)
    assert {c["state"] for c in cov["parity"]["cells"]} == {"missing"}
    cov = cm.build(catalogue([], witnesses=[witness]), DESIGNATIONS, HOLDS)
    assert {c["state"] for c in cov["parity"]["cells"]} == {"match"}


def test_dense_pixelized_interferometer_nufft_is_planned_at_sma_only():
    """Ruling R1(a) on #469: the dense pixelized NUFFT route is planned only at sma. The
    designations name it as the A100 secondary for the pixelized interferometer likelihoods."""
    dense = "jax_gpu_dense_fp64_nufft"
    for variant in ("rectangular", "delaunay"):
        assert dense in lc.planned_setups("interferometer", variant, "hpc_a100", "sma")
        for instrument in ("alma", "alma_high", "jvla"):
            planned = set(lc.planned_setups("interferometer", variant, "hpc_a100", instrument))
            assert dense not in planned and any(s.endswith("_wtilde") for s in planned)
        # Still designated: a measured dense row above sma is secondary, not experimental.
        assert lc.designation("interferometer", variant, "hpc_a100", dense) == "secondary"
    # The held datacube A100 group keeps the same scope (2f's D1, #475).
    assert dense in lc.planned_setups("datacube", "delaunay", "hpc_a100", "sma")
    for instrument in ("alma", "alma_high"):
        planned = set(lc.planned_setups("datacube", "delaunay", "hpc_a100", instrument))
        assert dense not in planned and any(s.endswith("_wtilde") for s in planned)
    # MGE is untouched.
    assert "jax_cpu_dense_fp64_nufft" in lc.planned_setups(
        "interferometer", "mge", "hpc_ral_cpu", "sma"
    )


def test_a_gpu_setup_is_planned_only_on_a_gpu_host():
    """A GPU-less host plans no GPU setup (the 2c/2e/2g planner gap)."""
    for row in DESIGNATIONS["designations"]:
        dataset_type, variant = row["likelihood"].split("/")
        planned = lc.planned_setups(dataset_type, variant, "hpc_ral_cpu", "sma")
        assert not [s for s in planned if s.split("_")[1] == "gpu"], row["id"]
    assert any(
        s.split("_")[1] == "gpu"
        for s in lc.planned_setups("point_source_source", "source_plane", "hpc_a100", "sma")
    )
