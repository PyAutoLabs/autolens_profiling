"""Semantic finding-ID and movable-anchor regression tests."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in sys.path:
    sys.path.insert(0, str(_misc))

from hazards._anchor import (  # noqa: E402
    CodeAnchor,
    _repo_path,
    normalized_tokens,
    token_fingerprint,
)
from hazards._measure import reachability_measurement  # noqa: E402
from hazards._record import (  # noqa: E402
    Finding,
    compare_finding_ids,
    findings_from_record,
    write_grouped_findings,
)
from hazards.aggregate import build_index_from_records  # noqa: E402


def _anchor(start_line: int = 2) -> CodeAnchor:
    tokens = normalized_tokens("value = minimum(value, 0.999)")
    return CodeAnchor(
        repo="Example",
        commit="a" * 40,
        path="example.py",
        start_line=start_line,
        end_line=start_line,
        token_fingerprint=token_fingerprint(tokens),
        tokens=tokens,
        symbol="example.clamp",
    )


def _finding(anchor: CodeAnchor | None = None) -> Finding:
    anchor = anchor or _anchor()
    return Finding(
        finding_id="component.example.semantic-clamp",
        title="Example clamp",
        summary="A stable semantic example.",
        hazard_class="saturation",
        tier=1,
        subject="component",
        subject_name="example",
        backends=("numpy",),
        measurements=(reachability_measurement(reachable_via=["numpy"]),),
        anchors=(anchor,),
        code_exists=True,
        reachable_via=("numpy",),
        blocked_by=(),
        affects_science=False,
    )


def test_anchor_classifies_unchanged_moved_changed_and_missing():
    anchor = _anchor(start_line=2)
    assert anchor.status_in_source("header\nvalue = minimum(value, 0.999)\n") == "unchanged"
    assert anchor.status_in_source("header\nother = 1\nvalue = minimum(value, 0.999)\n") == "moved"
    assert anchor.status_in_source("header\nvalue = minimum(value, 0.9)\n") == "changed"
    missing_line_anchor = _anchor(start_line=20)
    assert missing_line_anchor.status_in_source("header\nother = 1\n") == "missing"


def test_semantic_finding_id_does_not_depend_on_anchor_location():
    assert _finding(_anchor(2)).finding_id == _finding(_anchor(200)).finding_id


def test_new_persistent_and_resolved_ids_are_separate():
    diff = compare_finding_ids({"a.one", "b.two"}, {"b.two", "c.three"})
    assert diff.new == ("a.one",)
    assert diff.persistent == ("b.two",)
    assert diff.resolved == ("c.three",)
    assert diff.has_new


def test_grouped_record_round_trip_and_index_keying(tmp_path):
    finding = _finding()
    paths = write_grouped_findings([finding], tmp_path)
    assert len(paths) == 1
    assert findings_from_record(paths[0])[0]["finding_id"] == finding.finding_id

    index = build_index_from_records(tmp_path)
    assert list(index["findings"]) == [finding.finding_id]
    assert index["findings"][finding.finding_id]["record"] == ("component/example/saturation.json")


def test_record_writer_rejects_non_standard_nan_json(tmp_path):
    finding = replace(_finding(), reproducer={"gradient": float("nan")})
    with pytest.raises(ValueError, match="Out of range float"):
        write_grouped_findings([finding], tmp_path)


def test_source_anchor_checkout_path_in_grouped_and_flat_contexts(tmp_path):
    brain = tmp_path / "PyAutoBrain" / "agents"
    brain.mkdir(parents=True)
    (brain / "_repo_paths.py").write_text(
        "from pathlib import Path\n"
        "def repo_path(root, name, required=False):\n"
        "    flat = Path(root) / name\n"
        "    return flat if flat.is_dir() else next(p for p in Path(root).glob('*/' + name) if p.is_dir())\n"
    )
    galaxy = tmp_path / "galaxy" / "PyAutoGalaxy"
    galaxy.mkdir(parents=True)
    (galaxy / ".git").touch()
    assert _repo_path(tmp_path, "PyAutoGalaxy") == galaxy
    galaxy.rename(tmp_path / "PyAutoGalaxy")
    assert _repo_path(tmp_path, "PyAutoGalaxy") == tmp_path / "PyAutoGalaxy"


def test_source_anchor_uses_standalone_flat_checkout(tmp_path):
    galaxy = tmp_path / "PyAutoGalaxy"
    galaxy.mkdir()
    (galaxy / ".git").touch()
    assert _repo_path(tmp_path, "PyAutoGalaxy") == galaxy
