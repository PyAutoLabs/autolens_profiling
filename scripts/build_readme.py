"""
build_readme.py — refresh auto-generated tables in every README from the
latest artifacts under `results/`.

Run from the repo root:

    python scripts/build_readme.py            # rewrite README tables in place
    python scripts/build_readme.py --check    # exit non-zero if rewriting
                                              # would change any file (CI gate)

Each table region in a README is delimited by sentinel comments, e.g.

    <!-- BEGIN auto-table:runtime -->
    | ... |
    <!-- END auto-table:runtime -->

This script:

  1. Scans `results/{breakdown,simulators,searches}/**` for **versioned
     artifacts** (`<script>_<purpose>_<extras>_v<version>[_sparse].json`)
     and picks the latest version per group.
  2. Scans `results/runtime/<class>/<model>[/<instrument>]/comparison.json`
     for **sweep comparison artifacts** (written by
     `likelihood_runtime/aggregate.py`).
  3. When `results/baselines/<name>/` exists, reads the same comparison
     layout beneath it so dashboard tables can carry a named-baseline
     column (e.g. `PreOptimizationTimes`).
  4. Renders a markdown table per known region and replaces the content
     inside the matching sentinel block.

Regions covered today:

  - README.md                       | headline (runtime cells + breakdown)
  - likelihood_runtime/README.md    | runtime
  - likelihood_breakdown/README.md  | breakdown
  - simulators/README.md            | simulators
  - searches/README.md              | searches-nautilus

Artifact-shape reference: `results/notes/design_lock_in.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
RUNTIME_ROOT = RESULTS_ROOT / "runtime"
BASELINES_ROOT = RESULTS_ROOT / "baselines"

# Sentinel block: keeps surrounding hand-written prose intact, only the
# content between BEGIN and END is rewritten.
SENTINEL_RE = re.compile(
    r"(<!-- BEGIN auto-table:(?P<name>[a-z0-9_\-]+) -->)"
    r".*?"
    r"(<!-- END auto-table:(?P=name) -->)",
    re.DOTALL,
)

# Versioned artifact filename:
#   <script>_<purpose>_<extras>_v<version>[_sparse].json
# `<purpose>` is `summary` (runtime-style standalone artifacts, simulators,
# searches) or `breakdown` (likelihood_breakdown). `<extras>` is optional
# and captures the instrument / dataset_name suffix. Examples:
#   mge_breakdown_hst_v2026.5.29.4.json
#   pixelization_breakdown_hst_v2026.5.29.4_sparse.json
#   imaging_summary_v2026.5.14.2.json
#   simple_summary_v2026.5.14.2.json
ARTIFACT_RE = re.compile(
    r"^(?P<script>[a-z0-9_]+?)_(?P<purpose>summary|breakdown)"
    r"(?:_(?P<extra>[a-z0-9_]+?))?"
    r"_v(?P<version>[0-9]+(?:\.[0-9]+)+)"
    r"(?P<sparse>_sparse)?"
    r"\.json$"
)

# Canonical sweep-config column order (matches likelihood_runtime/aggregate.py).
CONFIG_ORDER = (
    "local_cpu_fp64",
    "local_cpu_mp",
    "local_gpu_fp64",
    "local_gpu_mp",
    "hpc_a100_fp64",
    "hpc_a100_mp",
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    section: str  # "breakdown", "simulators", "searches"
    subfolder: str  # "imaging", "nautilus", or "" for flat
    script: str  # e.g. "mge", "pixelization", "simple"
    purpose: str  # "summary" | "breakdown"
    instrument: str | None  # e.g. "hst", "sma", or None
    sparse: bool
    version: tuple[int, ...]
    raw_version: str

    @property
    def data(self) -> dict:
        return json.loads(self.path.read_text())


@dataclass(frozen=True)
class RuntimeCell:
    """One swept cell's comparison.json (class/model[/instrument])."""

    cell: tuple[str, ...]
    path: Path

    @property
    def cell_id(self) -> str:
        return "/".join(self.cell)

    @property
    def configs(self) -> dict:
        return json.loads(self.path.read_text()).get("configs", {})


def _parse_version(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split("."))


def _scan_artifacts() -> list[Artifact]:
    if not RESULTS_ROOT.exists():
        return []
    out: list[Artifact] = []
    for p in RESULTS_ROOT.rglob("*_v*.json"):
        rel = p.relative_to(RESULTS_ROOT).parts
        if len(rel) < 2 or rel[0] in ("runtime", "baselines"):
            continue
        section = rel[0]  # "breakdown" | "simulators" | "searches"
        subfolder = rel[1] if len(rel) > 2 else ""
        m = ARTIFACT_RE.match(p.name)
        if not m:
            continue
        out.append(
            Artifact(
                path=p,
                section=section,
                subfolder=subfolder,
                script=m["script"],
                purpose=m["purpose"],
                instrument=m["extra"],
                sparse=bool(m["sparse"]),
                version=_parse_version(m["version"]),
                raw_version=m["version"],
            )
        )
    return out


def _scan_runtime_cells(root: Path) -> list[RuntimeCell]:
    """Find every comparison.json under a runtime-layout root."""
    if not root.exists():
        return []
    cells = []
    for p in sorted(root.rglob("comparison.json")):
        cell = p.parent.relative_to(root).parts
        if 2 <= len(cell) <= 3:
            cells.append(RuntimeCell(cell=cell, path=p))
    return cells


def _baseline_names() -> list[str]:
    if not BASELINES_ROOT.exists():
        return []
    return sorted(d.name for d in BASELINES_ROOT.iterdir() if d.is_dir())


def _latest_per_group(artifacts: Iterable[Artifact], key) -> dict[tuple, Artifact]:
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


def _format_time(seconds: float | None) -> str:
    if seconds is None or not isinstance(seconds, (int, float)) or math.isnan(seconds):
        return "—"
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} μs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _config_headline_seconds(cfg: dict) -> float | None:
    """Per-call full-pipeline cost from one comparison.json config entry."""
    for key in (
        "full_pipeline_per_call",
        "full_pipeline_single_jit",
        "full_pipeline_cube_single_jit",
        "total_step_by_step_cube",
    ):
        v = cfg.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _config_vmap_seconds(cfg: dict) -> float | None:
    vmap = cfg.get("vmap")
    if isinstance(vmap, dict):
        v = vmap.get("per_call")
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _ordered_config_names(cells: list[RuntimeCell]) -> list[str]:
    """Canonical configs first, then any extras (e.g. *_sparse) seen in the data."""
    seen: list[str] = []
    for cell in cells:
        for name in cell.configs:
            if name not in seen:
                seen.append(name)
    ordered = [c for c in CONFIG_ORDER if c in seen]
    ordered += sorted(n for n in seen if n not in CONFIG_ORDER)
    return ordered


def _render_runtime_table(cells: list[RuntimeCell], baselines: dict[str, list[RuntimeCell]]) -> str:
    """Cells × configs matrix of full-pipeline per-call cost.

    When named baselines exist under ``results/baselines/``, one extra
    column per baseline shows that baseline's headline (first available
    config, preferring the A100 row) so regressions/improvements against
    e.g. ``PreOptimizationTimes`` are visible at a glance.
    """
    if not cells:
        return _no_data_block("run `likelihood_runtime/sweep.py` then `aggregate.py` to populate.")
    config_names = _ordered_config_names(cells)
    baseline_names = sorted(baselines)

    header = ["Cell"] + config_names + baseline_names
    rows = ["| " + " | ".join(header) + " |"]
    rows.append("|" + "|".join(["---"] * len(header)) + "|")

    baseline_by_cell = {
        name: {c.cell: c for c in cell_list} for name, cell_list in baselines.items()
    }

    def _headline_any_config(cell: RuntimeCell) -> float | None:
        cfgs = cell.configs
        for cname in reversed(_ordered_config_names([cell])):  # prefer A100/extras
            v = _config_headline_seconds(cfgs.get(cname, {}))
            if v is not None:
                return v
        return None

    for cell in cells:
        cfgs = cell.configs
        line = [f"`{cell.cell_id}`"]
        for cname in config_names:
            line.append(_format_time(_config_headline_seconds(cfgs.get(cname, {}))))
        for bname in baseline_names:
            bcell = baseline_by_cell[bname].get(cell.cell)
            line.append(_format_time(_headline_any_config(bcell)) if bcell else "—")
        rows.append("| " + " | ".join(line) + " |")
    return "\n" + "\n".join(rows) + "\n"


def _render_breakdown_table(artifacts: list[Artifact]) -> str:
    """One row per (class, script, instrument, path) with the step-sum total."""
    relevant = [a for a in artifacts if a.section == "breakdown" and a.purpose == "breakdown"]
    if not relevant:
        return _no_data_block("run a script under `likelihood_breakdown/` to populate.")
    latest = _latest_per_group(
        relevant, key=lambda a: (a.subfolder, a.script, a.instrument, a.sparse)
    )
    rows = ["| Cell | Instrument | Inversion path | Step-sum total | PyAutoLens version |"]
    rows.append("|------|------------|----------------|----------------|--------------------|")
    for (subfolder, script, instrument, sparse), art in sorted(latest.items()):
        total = art.data.get("total_step_by_step")
        rows.append(
            f"| `{subfolder}/{script}` | "
            f"{instrument or '—'} | "
            f"{'sparse (w-tilde)' if sparse else 'dense (mapping)'} | "
            f"{_format_time(total if isinstance(total, (int, float)) else None)} | "
            f"v{art.raw_version} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _simulator_total_seconds(art: Artifact) -> float | None:
    phases = art.data.get("phases")
    if isinstance(phases, dict):
        try:
            return float(sum(float(v) for v in phases.values()))
        except (TypeError, ValueError):
            return None
    return None


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
        rows.append(f"| `{script}.py` | {_format_time(total)} | v{art.raw_version} |")
    return "\n" + "\n".join(rows) + "\n"


def _render_nautilus_table(artifacts: list[Artifact]) -> str:
    relevant = [a for a in artifacts if a.section == "searches" and a.subfolder == "nautilus"]
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
        data = art.data
        perf = data.get("performance", {})
        conv = data.get("convergence", {})
        wall = _format_time(perf.get("wall_time_s"))
        per_eval = (
            f"{perf['time_per_eval_ms']:.1f} ms"
            if perf.get("time_per_eval_ms") is not None
            else "—"
        )
        evals_to_ml = f"{conv['evals_to_ml']:,}" if conv.get("evals_to_ml") is not None else "—"
        time_to_ml = _format_time(conv.get("time_to_ml_s"))
        rows.append(
            f"| `{script}.py` | {data.get('backend') or '—'} | "
            f"{wall} | {per_eval} | {evals_to_ml} | {time_to_ml} | "
            f"v{art.raw_version} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _render_headline(
    artifacts: list[Artifact],
    cells: list[RuntimeCell],
    baselines: dict[str, list[RuntimeCell]],
) -> str:
    """Top-level dashboard: runtime matrix + latest breakdown totals."""
    parts = ["\n**Likelihood runtime** — full-pipeline per-call cost per cell × config:\n"]
    parts.append(_render_runtime_table(cells, baselines).strip("\n"))
    parts.append("\n**Likelihood breakdown** — latest per-step decompositions:\n")
    parts.append(_render_breakdown_table(artifacts).strip("\n"))
    return "\n" + "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Region registry + rewrite driver
# ---------------------------------------------------------------------------


def _build_renderers():
    artifacts = _scan_artifacts()
    cells = _scan_runtime_cells(RUNTIME_ROOT)
    baselines = {name: _scan_runtime_cells(BASELINES_ROOT / name) for name in _baseline_names()}
    return artifacts, {
        "headline": lambda: _render_headline(artifacts, cells, baselines),
        "runtime": lambda: _render_runtime_table(cells, baselines),
        "breakdown": lambda: _render_breakdown_table(artifacts),
        "simulators": lambda: _render_simulator_table(artifacts),
        "searches-nautilus": lambda: _render_nautilus_table(artifacts),
    }


# Files that may contain auto-table regions. Listing them explicitly (rather
# than walking the repo) keeps the script's surface obvious.
TARGET_READMES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "likelihood_runtime" / "README.md",
    REPO_ROOT / "likelihood_breakdown" / "README.md",
    REPO_ROOT / "simulators" / "README.md",
    REPO_ROOT / "searches" / "README.md",
]


def _rewrite_file(path: Path, renderers: dict) -> tuple[str, str, list[str]]:
    """Return (original_text, rewritten_text, unknown_sentinels)."""
    original = path.read_text()
    unknown: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group("name")
        begin = match.group(1)
        end = match.group(3)
        renderer = renderers.get(name)
        if renderer is None:
            unknown.append(name)
            return match.group(0)  # leave intact
        return f"{begin}{renderer()}{end}"

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

    artifacts, renderers = _build_renderers()
    print(f"Scanned {len(artifacts)} versioned artifact(s) under {RESULTS_ROOT}")

    any_changed = False
    all_unknown: list[tuple[Path, str]] = []
    for target in TARGET_READMES:
        if not target.exists():
            print(f"  skip      {target.relative_to(REPO_ROOT)} — not present", flush=True)
            continue
        original, rewritten, unknown = _rewrite_file(target, renderers)
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
