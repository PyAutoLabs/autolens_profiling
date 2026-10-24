"""Coverage matrix: cell states, the size-fallback ruling, the stale rule, parity witnesses,
and the §3.3 machine copy agreeing with catalogue_lifecycle.py."""

from __future__ import annotations

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
    cov = cm.build(catalogue([record("a"), record("b", pin="2026.10.17.1")]), DESIGNATIONS)
    assert only(cov)["state"] == "current" and only(cov)["record"] == "a"
    cov = cm.build(catalogue([record("old", status="superseded")]), DESIGNATIONS)
    assert only(cov)["state"] == "superseded"
    cov = cm.build(catalogue([]), DESIGNATIONS)
    assert only(cov)["state"] == "missing" and only(cov)["reason"] == "no record"


def test_size_fallback_current_counts_as_missing():
    note = "size_fallback: no row at the standard 1250 source pixels exists for this key"
    cov = cm.build(catalogue([record("a", limitations=[note])]), DESIGNATIONS)
    assert only(cov)["state"] == "missing"
    assert only(cov)["reason"].startswith("size_fallback")
    assert cov["totals"]["missing_size_fallback"] == 1


def test_stale_is_a_pin_behind_the_catalogue_pin():
    # The catalogue pin is the newest record pin when the envelope declares none.
    newer = record("n", pin="2026.10.17.1", measurement="breakdown")
    planned = [cell(), cell(measurement="breakdown")]
    cov = cm.build(catalogue([record("a", pin="2026.8.17.1"), newer], planned), DESIGNATIONS)
    assert cov["catalogue_release_pin"] == "2026.10.17.1"
    assert cov["release_pin_source"] == "newest record"
    by = {c["measurement"]: c for c in cov["cells"]}
    assert by["runtime"]["state"] == "stale" and by["breakdown"]["state"] == "current"
    # Numeric, not lexical: 2026.10 is newer than 2026.9.
    cov = cm.build(catalogue([record("a", pin="2026.9.30.1"), newer], planned), DESIGNATIONS)
    assert {c["state"] for c in cov["cells"]} == {"stale", "current"}
    # An envelope pin wins; a record without a pin is stale.
    doc = catalogue([record("a", pin=None)])
    doc["release_pin"] = "2026.10.1.1"
    cov = cm.build(doc, DESIGNATIONS)
    assert cov["release_pin_source"] == "envelope" and only(cov)["state"] == "stale"


def test_local_current_never_covers_a_cell():
    doc = catalogue([record("a", host="local_gpu")], planned=[cell(host="local_gpu")])
    assert only(cm.build(doc, DESIGNATIONS))["state"] == "superseded"


def test_matrix_groups_likelihood_class_measurement():
    planned = [cell(), cell(setup_id="jax_gpu_sparse_fp64", cls="secondary")]
    cov = cm.build(catalogue([record("a")], planned), DESIGNATIONS)
    rows = {(r["likelihood"], r["class"], r["measurement"]): r for r in cov["matrix"]}
    assert rows[("imaging/delaunay", "optimal", "runtime")]["current"] == 1
    assert rows[("imaging/delaunay", "secondary", "runtime")]["missing"] == 1
    assert cov["totals"]["planned_cells"] == 2


def test_class_mismatch_against_the_designations_copy_is_reported():
    planned = [cell(setup_id="numba_cpu_sparse_fp64", host="hpc_ral_cpu", cls="secondary")]
    cov = cm.build(catalogue([], planned), DESIGNATIONS)
    [mismatch] = cov["class_mismatches"]
    assert mismatch["planned"] == "secondary" and mismatch["s3_3"] == "optimal"


def test_parity_states_per_setup_cell():
    planned = [cell(), cell(measurement="breakdown")]
    parity = record("p", measurement="parity")
    parity["parity"] = {"status": "diverged"}
    cov = cm.build(catalogue([parity], planned), DESIGNATIONS)
    assert cov["parity"]["totals"] == {
        "setup_cells": 1,
        "match": 0,
        "diverged": 1,
        "not_assessed": 0,
        "missing": 0,
    }
    witness = {**cell(), "status": "not_assessed", "location": "cell", "path": "x"}
    cov = cm.build(catalogue([], planned, [witness]), DESIGNATIONS)
    assert cov["parity"]["cells"][0]["state"] == "not_assessed"
    history = {**witness, "location": "history"}
    cov = cm.build(catalogue([], planned, [history]), DESIGNATIONS)
    assert cov["parity"]["cells"][0]["state"] == "missing"


def test_parity_records_are_not_unplanned():
    parity = record("p", measurement="parity")
    parity["parity"] = {"status": "match"}
    stray = record("s", setup_id="jax_cpu_na_fp64", host="local_laptop")
    cov = cm.build(catalogue([record("a"), parity, stray]), DESIGNATIONS)
    assert cov["unplanned_current_records"] == ["s"]
    assert cov["totals"]["current_records_unplanned"] == 1
    assert cov["parity"]["cells"][0]["state"] == "match"


def test_check_detects_drift(tmp_path, capsys):
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "catalogue").mkdir()
    (tmp_path / cm.CATALOGUE).write_text(json.dumps(catalogue([record("a")])))
    (tmp_path / cm.DESIGNATIONS).write_text(json.dumps(DESIGNATIONS))
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


def test_designations_copy_transcribes_every_s3_3_row():
    rows = DESIGNATIONS["rows"]
    assert len(rows) == 11
    for row in rows:
        assert set(row) >= {
            "likelihood",
            "gpu_a100",
            "cpu_many_core",
            "laptop",
            "secondary",
            "rule_branches_on",
            "routes",
        }
    assert DESIGNATIONS["source"]["section"].startswith("3.3")
    assert DESIGNATIONS["footnote"].startswith("Mixed precision is `experimental`")


def _routes():
    for row in DESIGNATIONS["rows"]:
        for route in row["routes"]:
            variants = route["variants"] if route["variants"] != "*" else ["mge"]
            for variant in variants:
                yield route["dataset_type"], variant


@pytest.mark.parametrize("dataset_type, variant", sorted(set(_routes())))
def test_designations_copy_agrees_with_catalogue_lifecycle(dataset_type, variant):
    every = {
        sid
        for t in DESIGNATIONS["tables"].values()
        for h in t.values()
        for v in h.values()
        for sid in v
    }
    every |= {s.replace("fp64", "mixed") for s in every}
    for host in ("hpc_a100", "hpc_ral_cpu", "local_laptop"):
        for sid in sorted(every):
            expected = lc.designation(dataset_type, variant, host, sid) or "experimental"
            assert cm.designated_class(DESIGNATIONS, dataset_type, variant, host, sid) == expected
    suffixes = tuple(DESIGNATIONS["regime_only_suffixes"])
    for host in ("hpc_a100", "hpc_ral_cpu"):
        planned = {
            sid: cm.designated_class(DESIGNATIONS, dataset_type, variant, host, sid)
            for sid in every
            if not sid.endswith(suffixes)
            and cm.designated_class(DESIGNATIONS, dataset_type, variant, host, sid)
            != "experimental"
        }
        assert planned == lc.planned_setups(dataset_type, variant, host)
