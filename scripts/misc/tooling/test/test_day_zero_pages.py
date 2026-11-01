"""Day-zero rule pages (build_day_zero_pages.py, autolens_profiling#501): generated from the
Pulse designation mirror, the holds, day_zero.json and the catalogue; never hand-edited."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

TOOLING = Path(__file__).resolve().parents[1]
REPO = TOOLING.parents[2]
spec = importlib.util.spec_from_file_location(
    "build_day_zero_pages", TOOLING / "build_day_zero_pages.py"
)
dz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dz)

DESIGNATIONS = json.loads((REPO / dz.DESIGNATIONS).read_text())["designations"]


def test_the_committed_pages_are_current():
    assert dz.main(["--root", str(REPO), "--check"]) == 0


def test_one_page_per_designated_likelihood_linking_every_record():
    outputs = dz.render(REPO)
    likelihoods = sorted({d["likelihood"] for d in DESIGNATIONS})
    assert len(likelihoods) == 20
    assert sorted(outputs) == sorted([dz.INDEX] + [dz.page_path(lk) for lk in likelihoods])
    for row in DESIGNATIONS:
        page = outputs[dz.page_path(row["likelihood"])]
        assert row["url"] in page and row["rule"] in page, row["id"]
        assert f"Optimal `{row['optimal']}`" in page
    for likelihood in likelihoods:
        page = outputs[dz.page_path(likelihood)]
        for section in ("## Rule", "## Day-zero evidence", "## Untested branches", "## Revisit"):
            assert section in page, (likelihood, section)
        assert "2026.10.25.1" in page and page.startswith(dz.STAMP)


def test_the_held_cell_is_named_and_carries_no_record_link():
    page = dz.render(REPO)[dz.page_path("datacube/delaunay")]
    assert "**held, not designated.**" in page
    assert "datacube-delaunay-standard-gpu-a100" not in page
    assert "the setups do not separate" in page
    assert "| held (provisional) |" in dz.render(REPO)[dz.INDEX]


def test_untested_branches_carry_the_rule_clause_and_the_derived_list():
    page = dz.render(REPO)[dz.page_path("interferometer/delaunay")]
    untested = page.split("## Untested branches", 1)[1].split("## Revisit", 1)[0]
    assert "Named in the `gpu_a100` rule: the GPU streaming route" in untested
    for phrase in (
        "Instruments: measured on `alma`, `alma_high`, `jvla`, `sma` only",
        "`source_pixels_1500`",
        "laptop and local GPUs have no designation",
        "Mixed precision: measured",
        "Parallel efficiency: projected throughput assumes 1.0",
    ):
        assert phrase in untested, phrase


def test_evidence_lists_only_stamped_runtime_records_with_resolving_links():
    outputs = dz.render(REPO)
    catalogue = json.loads((REPO / dz.CATALOGUE).read_text())
    stamped = {
        r["id"] for r in catalogue["records"] if r["day_zero"] and r["measurement"] == "runtime"
    }
    seen = set()
    for path, text in outputs.items():
        if path == dz.INDEX:
            continue
        for rid in stamped:
            if f"[`{rid}`]" in text:
                seen.add(rid)
        for target in [t.split(")")[0] for t in text.split("](")[1:]]:
            if not target.startswith("http"):
                assert (REPO / Path(path).parent / target).resolve().exists(), (path, target)
    assert seen == stamped


def test_check_detects_a_hand_edit(tmp_path):
    for rel in (dz.DESIGNATIONS, dz.HOLDS, dz.DAY_ZERO, dz.CATALOGUE):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    assert dz.main(["--root", str(tmp_path), "--check"]) == 1
    assert dz.main(["--root", str(tmp_path)]) == 0
    assert dz.main(["--root", str(tmp_path), "--check"]) == 0
    page = tmp_path / dz.page_path("weak/isothermal")
    page.write_text(page.read_text() + "hand edit\n")
    assert dz.main(["--root", str(tmp_path), "--check"]) == 1
