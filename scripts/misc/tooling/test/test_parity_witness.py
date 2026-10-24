"""The parity.json witness convention: what a valid witness is (fixtures only, no real values)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("parity_witness", TOOLING / "parity_witness.py")
pw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pw)


def witness(**extra):
    """A synthetic witness: the numbers are fixtures, not measurements."""
    doc = {
        "schema": "parity-witness",
        "version": 1,
        "status": "match",
        "log_likelihood": -100.0,
        "reference_log_likelihood": -100.004,
        "tolerance": 0.01,
        "reference": {"source": "fixture reference", "date": "2026-10-01"},
        "measured_at": "2026-10-02T03:04:05Z",
        "autolens_version": "2026.10.17.1",
    }
    doc.update(extra)
    return {k: v for k, v in doc.items() if v is not ...}


def test_valid_match_and_diverged():
    assert pw.validate(witness()) == []
    assert pw.validate(witness(status="diverged", reference_log_likelihood=-101.0)) == []
    assert pw.delta(witness()) == pytest.approx(0.004)


def test_status_must_agree_with_the_tolerance():
    [error] = pw.validate(witness(status="diverged"))
    assert "disagrees" in error
    [error] = pw.validate(witness(reference_log_likelihood=-99.0))
    assert "disagrees" in error


def test_not_assessed_needs_a_reason_and_no_numbers():
    doc = witness(
        status="not_assessed",
        log_likelihood=None,
        reference_log_likelihood=None,
        tolerance=None,
    )
    assert any("reason" in e for e in pw.validate(doc))
    assert pw.validate({**doc, "reason": "reference run pending"}) == []


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"status": "ok"}, "status"),
        ({"schema": "other"}, "schema"),
        ({"reference": {"date": "2026-10-01"}}, "reference.source"),
        ({"reference": {"source": "x", "date": "yesterday"}}, "reference.date"),
        ({"measured_at": "2026-10-02 03:04"}, "measured_at"),
        ({"autolens_version": ...}, "autolens_version"),
        ({"tolerance": -1.0}, "non-negative"),
        ({"log_likelihood": float("nan")}, "finite"),
        ({"log_likelihood": None}, "finite"),
        ({"source_pixels": 0}, "source_pixels"),
    ],
)
def test_invalid_witnesses_are_refused(change, needle):
    errors = pw.validate(witness(**change))
    assert any(needle in e for e in errors), errors


def test_check_validates_every_witness_in_results(tmp_path, capsys):
    cell = tmp_path / "results/imaging/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu"
    cell.mkdir(parents=True)
    (cell / "parity.json").write_text(json.dumps(witness()))
    assert pw.main(["--root", str(tmp_path), "--check"]) == 0
    (cell / "parity.json").write_text(json.dumps(witness(status="diverged")))
    assert pw.main(["--root", str(tmp_path), "--check"]) == 1
    assert "disagrees" in capsys.readouterr().out
