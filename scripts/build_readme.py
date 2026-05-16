"""
build_readme.py — refresh auto-generated tables in every README from the
latest versioned artifacts under `results/`.

Run from the repo root:

    python scripts/build_readme.py            # rewrite README tables in place
    python scripts/build_readme.py --check    # exit non-zero if rewriting
                                              # would change any file (CI gate)

Each table region in a README is delimited by sentinel comments, e.g.

    <!-- BEGIN auto-table:likelihood-imaging -->
    | ... |
    <!-- END auto-table:likelihood-imaging -->

This script:

  1. Scans `results/**/*_summary_v<version>.json`.
  2. Parses filenames into (section, sub-folder, script, instrument, version).
  3. Picks the latest version per group via PEP 440-ish dotted-version sort.
  4. Generates a markdown table per known region type and replaces the
     content inside the matching sentinel block.

Sections covered today:

  - top-level README.md
      <!-- BEGIN auto-table:headline --> ... <!-- END auto-table:headline -->
  - likelihood/README.md (section overview)
  - likelihood/imaging/README.md      | likelihood-imaging
  - likelihood/interferometer/README.md | likelihood-interferometer
  - likelihood/point_source/README.md | likelihood-point_source
  - likelihood/datacube/README.md     | likelihood-datacube
  - simulators/README.md              | simulators
  - searches/nautilus/README.md       | searches-nautilus

Hardware-tier columns (CPU / laptop GPU / HPC GPU) are deferred — every
artifact today is implicitly CPU and the table shows a single "Latest"
column. Once future artifacts encode hardware in the filename or JSON
(`*_summary_v<version>_<hardware>.json` or `{"hardware": "a100"}`), the
column logic in `_render_*_table` will be extended without touching the
sentinel layout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

# Sentinel block: keeps surrounding hand-written prose intact, only the
# content between BEGIN and END is rewritten.
SENTINEL_RE = re.compile(
    r"(<!-- BEGIN auto-table:(?P<name>[a-z0-9_\-]+) -->)"
    r".*?"
    r"(<!-- END auto-table:(?P=name) -->)",
    re.DOTALL,
)

# Artifact filename: <script>_summary_<extras>_v<version>.json
# `<extras>` is optional and captures the instrument / dataset_name suffix
# used by likelihood/imaging, likelihood/interferometer, likelihood/datacube,
# likelihood/point_source variants. Examples:
#   mge_likelihood_summary_hst_v2026.5.14.2.json
#   image_plane_summary_v2026.5.14.2.json
#   delaunay_likelihood_summary_sma_v2026.5.14.2.json
#   imaging_summary_v2026.5.14.2.json
#   simple_summary_v2026.5.14.2.json
ARTIFACT_RE = re.compile(
    r"^(?P<script>[a-z0-9_]+?)_summary"
    r"(?:_(?P<extra>[a-z0-9_]+?))?"
    r"_v(?P<version>[0-9]+(?:\.[0-9]+)+)"
    r"\.json$"
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    section: str  # "likelihood", "simulators", "searches"
    subfolder: str  # "imaging", "interferometer", "nautilus", or "" for flat
    script: str  # e.g. "mge", "image_plane", "simple"
    instrument: Optional[str]  # e.g. "hst", "sma", or None for simulators
    version: tuple[int, ...]
    raw_version: str

    @property
    def data(self) -> dict:
        return json.loads(self.path.read_text())


def _parse_version(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split("."))


def _scan_artifacts() -> list[Artifact]:
    if not RESULTS_ROOT.exists():
        return []
    out: list[Artifact] = []
    for p in RESULTS_ROOT.rglob("*_summary*_v*.json"):
        rel = p.relative_to(RESULTS_ROOT).parts
        if len(rel) < 2:
            continue
        section = rel[0]  # "likelihood" | "simulators" | "searches"
        subfolder = rel[1] if len(rel) > 2 else ""
        m = ARTIFACT_RE.match(p.name)
        if not m:
            continue
        # The "extra" group is the instrument label for likelihood scripts
        # that profile a single instrument (mge / pixelization / delaunay /
        # image_plane on hst, sma, etc.). For simulators and searches, the
        # filename has no extras and `extra` is None.
        script_name = m["script"].replace("_likelihood", "")
        out.append(
            Artifact(
                path=p,
                section=section,
                subfolder=subfolder,
                script=script_name,
                instrument=m["extra"],
                version=_parse_version(m["version"]),
                raw_version=m["version"],
            )
        )
    return out


def _latest_per_group(
    artifacts: Iterable[Artifact], key
) -> dict[tuple, Artifact]:
    """For each group key, keep the artifact with the highest version."""
    latest: dict[tuple, Artifact] = {}
    for a in artifacts:
        k = key(a)
        if k not in latest or a.version > latest[k].version:
            latest[k] = a
    return latest


# ---------------------------------------------------------------------------
# Per-region table rendering
# ---------------------------------------------------------------------------


def _no_data_block(message: str) -> str:
    return f"\n_No data yet — {message}_\n"


def _format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} μs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _likelihood_headline_seconds(art: Artifact) -> Optional[float]:
    """Steady-state per-call cost for the full-pipeline single JIT.

    Robust to the slight key-shape variation across the imaging /
    interferometer / point_source / datacube JSON layouts.
    """
    data = art.data
    # Imaging mge/pixelization/delaunay JSON shape: top-level key per-step
    # plus aggregates at end; the full-pipeline number is under "summary"
    # in some scripts and at top-level in others. Try several keys.
    for path in (
        ("summary", "full_pipeline_single_jit_s"),
        ("summary", "full_pipeline_s"),
        ("aggregate", "full_pipeline_single_jit_s"),
        ("full_pipeline_single_jit_s",),
        ("full_pipeline_s",),
    ):
        node = data
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, (int, float)):
            return float(node)
    return None


def _simulator_total_seconds(art: Artifact) -> Optional[float]:
    data = art.data
    phases = data.get("phases")
    if isinstance(phases, dict):
        try:
            return float(sum(float(v) for v in phases.values()))
        except (TypeError, ValueError):
            return None
    return None


def _nautilus_headline(art: Artifact) -> dict:
    data = art.data
    perf = data.get("performance", {})
    conv = data.get("convergence", {})
    return {
        "wall_time_s": perf.get("wall_time_s"),
        "time_per_eval_ms": perf.get("time_per_eval_ms"),
        "evals_to_ml": conv.get("evals_to_ml"),
        "time_to_ml_s": conv.get("time_to_ml_s"),
        "backend": data.get("backend"),
    }


def _render_likelihood_section_table(
    artifacts: list[Artifact], subfolder: str
) -> str:
    """One row per (script, instrument) pair for a single likelihood subfolder."""
    relevant = [
        a for a in artifacts if a.section == "likelihood" and a.subfolder == subfolder
    ]
    if not relevant:
        return _no_data_block(
            "run a script under this folder to populate. See section README."
        )
    latest = _latest_per_group(relevant, key=lambda a: (a.script, a.instrument))
    rows = ["| Script | Instrument | Latest single-JIT per-call | PyAutoLens version |"]
    rows.append("|--------|------------|----------------------------|--------------------|")
    for (script, instrument), art in sorted(latest.items()):
        seconds = _likelihood_headline_seconds(art)
        rows.append(
            f"| `{script}.py` | "
            f"{instrument or '—'} | "
            f"{_format_time(seconds)} | "
            f"v{art.raw_version} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _render_simulator_table(artifacts: list[Artifact]) -> str:
    relevant = [a for a in artifacts if a.section == "simulators"]
    if not relevant:
        return _no_data_block(
            "run a simulator under `simulators/` to populate. See section README."
        )
    latest = _latest_per_group(relevant, key=lambda a: a.script)
    rows = ["| Script | Total wall time | PyAutoLens version |"]
    rows.append("|--------|-----------------|--------------------|")
    for script, art in sorted(latest.items()):
        total = _simulator_total_seconds(art)
        rows.append(
            f"| `{script}.py` | {_format_time(total)} | v{art.raw_version} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _render_nautilus_table(artifacts: list[Artifact]) -> str:
    relevant = [
        a for a in artifacts if a.section == "searches" and a.subfolder == "nautilus"
    ]
    if not relevant:
        return _no_data_block(
            "run `searches/nautilus/{simple,jax}.py` to populate. See section README."
        )
    latest = _latest_per_group(relevant, key=lambda a: a.script)
    rows = [
        "| Script | Backend | Wall time | Time / eval | Evals → ML | Time → ML | PyAutoLens version |"
    ]
    rows.append(
        "|--------|---------|-----------|-------------|-----------|-----------|--------------------|"
    )
    for script, art in sorted(latest.items()):
        h = _nautilus_headline(art)
        wall = _format_time(h["wall_time_s"])
        per_eval = (
            f"{h['time_per_eval_ms']:.1f} ms"
            if h["time_per_eval_ms"] is not None
            else "—"
        )
        evals_to_ml = (
            f"{h['evals_to_ml']:,}" if h["evals_to_ml"] is not None else "—"
        )
        time_to_ml = _format_time(h["time_to_ml_s"])
        rows.append(
            f"| `{script}.py` | {h['backend'] or '—'} | "
            f"{wall} | {per_eval} | {evals_to_ml} | {time_to_ml} | "
            f"v{art.raw_version} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _render_headline_table(artifacts: list[Artifact]) -> str:
    """Top-level cross-section instrument × model headline.

    Rows are (section, subfolder, instrument); columns are scripts. Today
    likelihood/ has the richest cross-product; simulators are single-row
    per script. Build a compact 'latest result per axis' table.
    """
    likelihood = [a for a in artifacts if a.section == "likelihood"]
    if not likelihood:
        return _no_data_block(
            "run likelihood scripts to populate. See `likelihood/README.md`."
        )
    latest = _latest_per_group(
        likelihood, key=lambda a: (a.subfolder, a.script, a.instrument)
    )
    rows = [
        "| Section | Script | Instrument | Latest single-JIT per-call | PyAutoLens version |"
    ]
    rows.append(
        "|---------|--------|------------|----------------------------|--------------------|"
    )
    for (subfolder, script, instrument), art in sorted(latest.items()):
        seconds = _likelihood_headline_seconds(art)
        rows.append(
            f"| likelihood/{subfolder} | `{script}.py` | "
            f"{instrument or '—'} | "
            f"{_format_time(seconds)} | "
            f"v{art.raw_version} |"
        )
    return "\n" + "\n".join(rows) + "\n"


# Registry mapping sentinel name → renderer
RENDERERS = {
    "headline": _render_headline_table,
    "likelihood-imaging": lambda arts: _render_likelihood_section_table(arts, "imaging"),
    "likelihood-interferometer": lambda arts: _render_likelihood_section_table(
        arts, "interferometer"
    ),
    "likelihood-point_source": lambda arts: _render_likelihood_section_table(
        arts, "point_source"
    ),
    "likelihood-datacube": lambda arts: _render_likelihood_section_table(
        arts, "datacube"
    ),
    "simulators": _render_simulator_table,
    "searches-nautilus": _render_nautilus_table,
}


# Files that may contain auto-table regions. Listing them explicitly (rather
# than walking the repo) keeps the script's surface obvious and prevents
# accidental rewrites of e.g. workspace_developer mirror docs that may end
# up here later.
TARGET_READMES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "likelihood" / "README.md",
    REPO_ROOT / "likelihood" / "imaging" / "README.md",
    REPO_ROOT / "likelihood" / "interferometer" / "README.md",
    REPO_ROOT / "likelihood" / "point_source" / "README.md",
    REPO_ROOT / "likelihood" / "datacube" / "README.md",
    REPO_ROOT / "simulators" / "README.md",
    REPO_ROOT / "searches" / "README.md",
    REPO_ROOT / "searches" / "nautilus" / "README.md",
]


def _rewrite_file(
    path: Path, artifacts: list[Artifact]
) -> tuple[str, str, list[str]]:
    """Return (original_text, rewritten_text, unknown_sentinels)."""
    original = path.read_text()
    unknown: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group("name")
        begin = match.group(1)
        end = match.group(3)
        renderer = RENDERERS.get(name)
        if renderer is None:
            unknown.append(name)
            return match.group(0)  # leave intact
        body = renderer(artifacts)
        return f"{begin}{body}{end}"

    rewritten = SENTINEL_RE.sub(replace, original)
    return original, rewritten, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any target file would be rewritten (CI gate).",
    )
    args = parser.parse_args(argv)

    artifacts = _scan_artifacts()
    print(f"Scanned {len(artifacts)} artifact(s) under {RESULTS_ROOT}")

    any_changed = False
    all_unknown: list[tuple[Path, str]] = []
    for target in TARGET_READMES:
        if not target.exists():
            print(f"  skip      {target.relative_to(REPO_ROOT)} — not present", flush=True)
            continue
        original, rewritten, unknown = _rewrite_file(target, artifacts)
        for u in unknown:
            all_unknown.append((target, u))
        if rewritten == original:
            print(f"  unchanged {target.relative_to(REPO_ROOT)}", flush=True)
            continue
        any_changed = True
        if args.check:
            print(f"  WOULD rewrite {target.relative_to(REPO_ROOT)}", flush=True)
        else:
            target.write_text(rewritten)
            print(f"  rewrote   {target.relative_to(REPO_ROOT)}", flush=True)

    for path, name in all_unknown:
        print(
            f"WARNING: unknown sentinel '{name}' in {path.relative_to(REPO_ROOT)} — left intact",
            file=sys.stderr,
        )

    if args.check and any_changed:
        print(
            "ERROR: `build_readme.py --check` found pending changes. "
            "Run `python scripts/build_readme.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
