"""Render the scientist-facing setup browser using Brain's shared board theme."""

from __future__ import annotations

import html
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[3]
REPO = "https://github.com/PyAutoLabs/autolens_profiling"


def theme():
    """Resolve the real Brain in supported flat/worktree and grouped layouts."""
    candidates = []
    if os.environ.get("PYAUTO_BRAIN"):
        candidates.append(Path(os.environ["PYAUTO_BRAIN"]))
    candidates.extend((ROOT.parent / "PyAutoBrain", ROOT.parent.parent / "organs/PyAutoBrain"))
    for candidate in candidates:
        path = candidate / "board/_theme.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("profiling_board_theme", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(
        "Shared board theme missing: check out PyAutoBrain beside this repo or set PYAUTO_BRAIN."
    )


def source_url(path, revision):
    if (
        not isinstance(path, str)
        or path.startswith("/")
        or any(p in ("", ".", "..") for p in path.split("/"))
    ):
        raise ValueError("Invalid evidence path")
    return f"{REPO}/blob/{quote(revision or 'main', safe='')}/{quote(path, safe='/')}"


def render(doc):
    shared = theme()
    escape = html.escape
    fallback = []
    for setup in doc["setups"]:
        ev = setup.get("evidence")
        if ev:
            fallback.append(
                f'<li><a href="{escape(source_url(ev["path"], doc.get("producer_revision")), quote=True)}">'
                f"{escape(setup['dataset'])} / {escape(setup['model'])} / {escape(setup.get('instrument') or 'unspecified')} — "
                f"{escape(ev['path'])}{escape(ev.get('fragment') or '')}</a></li>"
            )
    css = (ROOT / "catalogue/browser.css").read_text()
    js = (ROOT / "catalogue/browser.js").read_text()
    # The content hash binds the initial catalogue to this generated page too.
    import hashlib

    serialized = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    sha = hashlib.sha256(serialized.encode()).hexdigest()
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>AutoLens profiling — browse by setup</title>"
        f"<style>{shared.css('pulse')}\n{css}</style></head><body>"
        '<a class="skip" href="#browser">Skip to setups</a>'
        + shared.hero("pulse", "AutoLens · Profiling")
        + '<main><div class="intro"><h2>Find your profiling setup</h2>'
        "<p>Choose a dataset and model, then inspect the instrument and configuration that match your work.</p>"
        '<p class="qualification">Archive evidence · new baseline pending</p>'
        '<p class="muted">Recorded results remain unreviewed. Missing measurements stay visible.</p></div>'
        '<nav class="quick-links" aria-label="Profiling resources">'
        '<a href="https://pyautolabs.github.io/PyAutoPulse/">Profiling campaigns ↗</a>'
        f'<a href="{REPO}/blob/main/wiki/index.md">Research wiki ↗</a>'
        f'<a href="{REPO}/blob/main/catalogue/README.md">Evidence policy ↗</a></nav>'
        f'<section id="browser" data-catalogue-sha="{sha}" aria-label="Choose a profiling setup">'
        '<p id="load-status" role="status" aria-live="polite">Loading setup catalogue…</p>'
        '<div id="navigation"></div><section id="results" aria-label="Selected setup" hidden></section></section>'
        "<noscript><p>Enable JavaScript for selectors and charts. Original evidence is available below.</p>"
        "<details><summary>Browse original profiling evidence</summary><ul>"
        + "".join(fallback)
        + '</ul></details></noscript></main><footer><p class="muted">'
        "Timings, compilation and memory describe their recorded setup. An unmeasured cell is not a zero.</p></footer>"
        f"<script>{js}</script></body></html>\n"
    )
