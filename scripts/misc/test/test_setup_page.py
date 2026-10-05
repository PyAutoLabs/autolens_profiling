"""Page publication contract and hostile evidence text handling."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1] / "tooling"
spec = importlib.util.spec_from_file_location("setup_page", TOOLING / "setup_page.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)


def test_page_binds_exact_catalogue_bytes_and_shared_theme():
    doc = json.loads((TOOLING.parents[2] / "dashboard/catalogue.json").read_text())
    rendered = page.render(doc)
    data = json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"
    assert hashlib.sha256(data.encode()).hexdigest() in rendered
    assert page.theme().hero("pulse", "AutoLens · Profiling") in rendered
    assert page.theme().css("pulse") in rendered
    assert 'id="navigation"' in rendered
    assert "<noscript>" in rendered
    assert "series.json" not in rendered
    assert "run-time-over-time" not in rendered


def test_no_script_evidence_is_escaped_and_pinned():
    doc = {
        "producer_revision": "a" * 40,
        "setups": [
            {
                "dataset": '<img src=x onerror="bad()">',
                "model": "x&y",
                "instrument": None,
                "evidence": {
                    "path": 'results/a"b.json',
                    "fragment": "/x</a><script>bad()</script>",
                },
            }
        ],
    }
    rendered = page.render(doc)
    assert "<img src=x" not in rendered
    assert "<script>bad()" not in rendered
    assert "&lt;img" in rendered
    assert "/blob/" + "a" * 40 + "/results/a%22b.json" in rendered


@pytest.mark.parametrize("path", ["/outside", "../outside", "a/../b", "a//b"])
def test_fallback_refuses_unsafe_evidence(path):
    with pytest.raises(ValueError, match="Invalid evidence"):
        page.source_url(path, "a" * 40)
