"""
build_readme.py — refresh auto-generated tables in every README from the
latest artifacts under `results/`.

Run from the repo root:

    python scripts/misc/tooling/build_readme.py         # rewrite README tables in place
    python scripts/misc/tooling/build_readme.py --check # exit non-zero if rewriting
                                              # would change any file (CI gate)

Each table region in a README is delimited by sentinel comments, e.g.

    <!-- BEGIN auto-table:runtime -->
    | ... |
    <!-- END auto-table:runtime -->

This script:

  1. Reads the mirrored results tree natively (autolens_profiling#442,
     `results/README.md`), keyed on the v3 identity the catalogue builder uses:
     - breakdown rows: every current `results/<dataset_type>/<variant>/
       <instrument>/<setup_id>/<host_class>[/<model_variant>]/breakdown.json`,
       then every published row with no v3 cell, latest per pre-move group,
       read where the move put it under the name it kept: unresolved rows in
       `results/breakdown/`, `results/misc/likelihood_breakdown/` and
       `archive/YYYY/MM/<slug>/breakdown/` (labelled archived);
     - sweep comparisons: `results/<dataset_type>/<variant>/<instrument>/
       comparison.json` (written by `likelihood_runtime/aggregate.py`) and the
       cells with no derivable v3 identity under
       `results/misc/likelihood_runtime/**/comparison.json`.
  2. Scans `results/lens/**` and `results/misc/{simulators,pipeline_resume}/**`
     for **versioned artifacts** (`<script>_<purpose>_<extras>_v<version>
     [_sparse].json`) and picks the latest version per group; the
     simulator and pipeline-resume evidence measured before the day-zero move
     is read from its archive folder (`archive/YYYY/MM/<slug>/<group>/`) and
     labelled archived.
  3. Archived named baselines (`archive/YYYY/MM/<name>/<cell>/comparison.json`,
     e.g. `PreOptimizationTimes`) add a column to the runtime table, matched to
     the current cell through the cell map in `catalogue/series_id_map.json`
     (generated from the day-zero migration map).
  4. Renders a markdown table per known region and replaces the content
     inside the matching sentinel block.

Regions covered today:

  - README.md                       | headline (runtime cells + breakdown)
  - likelihood_runtime/README.md    | runtime
  - likelihood_breakdown/README.md  | breakdown
  - simulators/README.md            | simulators
  - hazards/README.md               | hazards
  - lens/deflections/README.md      | deflections
  - lens/solver/README.md           | solver-corpus, solver-accuracy, solver-early-stopping,
                                    | solver-accuracy-posthoc, solver-euclid-latent,
                                    | solver-timing

Artifact-shape reference: `wiki/campaigns/misc/design_lock_in.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalogue_lifecycle as lc  # noqa: E402
from build_dashboard import (  # noqa: E402
    ESTIMATOR_LEGACY,
    ESTIMATOR_MEDIAN,
    SERIES_ID_MAP,  # noqa: E402
    headline_median_seconds,
    marker_verdict,
    read_release_sweep_conf,
    repeat_summary,
)

_RELEASE_SWEEP_CONF = read_release_sweep_conf(REPO_ROOT)

RESULTS_ROOT = REPO_ROOT / "results"
ARCHIVE_ROOT = REPO_ROOT / "archive"
#: The pre-move breakdown folder, as the day-zero map records old paths (read, never written).
LEGACY_BREAKDOWN_PREFIX = "results/breakdown/"
#: The pre-#442 breakdown folder under a results root, read for its unresolved rows only.
LEGACY_BREAKDOWN_DIR = "breakdown"
#: results/misc/<group>/ folders whose versioned artifacts feed a README table.
ARTIFACT_GROUPS = ("simulators", "pipeline_resume")

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
# `<purpose>` is `summary` (runtime-style standalone artifacts, simulators)
# or `breakdown` (likelihood_breakdown). `<extras>` is optional
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

# Config-tagged artifact filename (sweep-style `--config-name` outputs, e.g.
# the A100 tier): <script>_<config>[_sparse].json. Instrument and version are
# not in the name — they are read from the JSON payload instead. Only the
# unresolved rows the day-zero move left under results/breakdown/ still carry it.
#   delaunay_hpc_a100_fp64.json
CONFIG_TAGGED_RE = re.compile(
    r"^(?P<script>[a-z0-9_]+?)_(?P<config>local_cpu_fp64|local_cpu_mp|"
    r"local_gpu_fp64|local_gpu_mp|hpc_ral_cpu_fp64|hpc_a100_fp64|hpc_a100_mp)"
    r"(?P<sparse>_sparse)?"
    r"\.json$"
)

# Canonical sweep-config column order (matches likelihood_runtime/aggregate.py).
CONFIG_ORDER = (
    "local_cpu_fp64",
    "local_cpu_mp",
    "local_gpu_fp64",
    "local_gpu_mp",
    # RAL CPU-only rows (point-source CPU campaign, autolens_profiling#297).
    "hpc_ral_cpu_fp64",
    "hpc_a100_fp64",
    "hpc_a100_mp",
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    section: str  # "breakdown", "simulators", "lens"
    subfolder: str  # "imaging", "nautilus", or "" for flat
    script: str  # e.g. "mge", "pixelization", "simple"
    purpose: str  # "summary" | "breakdown"
    instrument: str | None  # e.g. "hst", "sma", or None
    sparse: bool
    version: tuple[int, ...]
    raw_version: str
    # Sweep config for config-tagged artifacts; untagged (single-config
    # filename pattern) runs are local CPU fp64 by package convention.
    config: str = "local_cpu_fp64"
    # Read from archive/ (measured before the day-zero move, never current).
    archived: bool = False

    @property
    def version_label(self) -> str:
        return f"v{self.raw_version}" + (" (archived)" if self.archived else "")

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


def _artifact(path: Path, section: str, subfolder: str, archived: bool = False) -> Artifact | None:
    m = ARTIFACT_RE.match(path.name)
    if m is None:
        return None
    return Artifact(
        path=path,
        section=section,
        subfolder=subfolder,
        script=m["script"],
        purpose=m["purpose"],
        instrument=m["extra"],
        sparse=bool(m["sparse"]),
        version=_parse_version(m["version"]),
        raw_version=m["version"],
        archived=archived,
    )


def _scan_group(base: Path, section: str, archived: bool = False) -> list[Artifact]:
    """Versioned artifacts under one group folder (``subfolder`` = its first sub-folder)."""
    out = []
    for p in sorted(base.rglob("*.json")) if base.is_dir() else []:
        rel = p.relative_to(base).parts
        art = _artifact(p, section, rel[0] if len(rel) > 1 else "", archived)
        if art is not None:
            out.append(art)
    return out


def _scan_artifacts() -> list[Artifact]:
    """Versioned artifacts: ``results/lens/``, the misc groups, and their archived history."""
    out = _scan_group(RESULTS_ROOT / "lens", "lens")
    for group in ARTIFACT_GROUPS:
        out += _scan_group(RESULTS_ROOT / "misc" / group, group)
        # archive/YYYY/MM/<slug>/<group>/...: the slug folder keeps the pre-move path.
        for base in sorted(ARCHIVE_ROOT.glob(f"*/*/*/{group}")):
            out += _scan_group(base, group, archived=True)
    return out


@dataclass(frozen=True)
class BreakdownRow:
    """One current breakdown row, keyed on its v3 identity.

    ``setup`` / ``host`` are the v3 ``setup_id`` / ``host_class``; an unresolved row the
    day-zero move left under ``results/breakdown/`` (setup axis never recorded, pending a
    human ruling) has no setup and names its sweep config label as the host.
    """

    path: Path
    cell: str
    instrument: str | None
    setup: str | None
    host: str
    sparse: bool
    #: The release the row's file name records (versioned artifacts); else read from the JSON.
    version: str | None = None
    archived: bool = False

    @property
    def data(self) -> dict:
        return json.loads(self.path.read_text())

    @property
    def version_label(self) -> str:
        version = self.version or self.data.get("autolens_version")
        if not version:
            return "—"
        return f"v{version}" + (" (archived)" if self.archived else "")


#: Host order in the table (the laptop first, the reference hosts last, as CONFIG_ORDER).
HOST_ORDER = ("local_laptop", "local_gpu", "hpc_ral_cpu", "hpc_a100") + CONFIG_ORDER


def _scan_breakdown_rows(results_root: Path = RESULTS_ROOT) -> list[BreakdownRow]:
    """Every current mirrored ``breakdown.json`` cell, then the unresolved legacy rows."""
    root = results_root.parent
    rows = []
    for p in sorted(results_root.rglob("breakdown.json")) if results_root.is_dir() else []:
        f = lc.mirrored(p.relative_to(root).as_posix())
        if f is None or f["location"] != "cell":
            continue
        mv = f"/{f['model_variant']}" if f["model_variant"] != "default" else ""
        mv += f"/{f['size_variant']}" if f["size_variant"] else ""
        rows.append(
            BreakdownRow(
                path=p,
                cell=f"{f['dataset_type']}/{f['variant']}{mv}",
                instrument=f["instrument"],
                setup=f["setup_id"],
                host=f["host_class"],
                sparse=f["setup"]["operator"] == "sparse",
            )
        )
    return rows + _legacy_breakdown_rows(results_root)


def _legacy_breakdown_artifact(path: Path, rel: tuple[str, ...]):
    """``(group, version, row)`` of a breakdown file under its pre-move name ``rel``.

    ``path`` is where the file lives now; ``rel`` its path below the old ``results/breakdown/``
    (whose last part is the name the grammar reads); the grammar is the pre-move
    table's (versioned ``*_breakdown_*_v<version>`` artifacts, config-tagged rows with an
    ``autolens_version``), grouped as it grouped them.
    """
    name = rel[-1]
    m = ARTIFACT_RE.match(name)
    config, version = "local_cpu_fp64", None
    if m is not None:
        if m["purpose"] != "breakdown":
            return None
        version, instrument = m["version"], m["extra"]
    else:
        m = CONFIG_TAGGED_RE.match(name)
        if m is None:
            return None
        try:
            payload = json.loads(path.read_text())
            version = str(payload.get("autolens_version", ""))
            _parse_version(version)
        except (OSError, ValueError):
            return None
        config, instrument = m["config"], payload.get("instrument")
    instrument = instrument if isinstance(instrument, str) else None
    subfolder = rel[0] if len(rel) > 1 else ""
    group = (subfolder, m["script"], instrument, bool(m["sparse"]), config)
    row = BreakdownRow(
        path=path,
        cell=f"{subfolder}/{m['script']}",
        instrument=instrument,
        setup=None,
        host=config,
        sparse=bool(m["sparse"]),
        version=version,
    )
    return group, _parse_version(version), row


def _legacy_breakdown_rows(results_root: Path = RESULTS_ROOT) -> list[BreakdownRow]:
    """Published breakdown rows whose file has no current v3 cell, latest per pre-move group.

    Read where the day-zero move put them, under the name they kept: the unresolved rows left
    in ``results/breakdown/`` (setup never recorded), ``results/misc/likelihood_breakdown/``
    (no derivable cell; current) and ``archive/YYYY/MM/<slug>/breakdown/`` (labelled
    archived). A group whose newest release the map moved into a mirrored cell is shown by
    that cell's v3 row instead (``catalogue/migration_day_zero.json`` names the file's old
    name), so a superseded archived release never reappears.
    """
    root = results_root.parent
    sources = [
        (results_root / LEGACY_BREAKDOWN_DIR, False),
        (results_root / "misc" / "likelihood_breakdown", False),
    ]
    sources += [(b, True) for b in sorted((root / "archive").glob("*/*/*/breakdown"))]
    latest: dict[tuple, tuple] = {}

    def offer(found, mirrored=False):
        if found is not None:
            group, version, row = found
            if group not in latest or version > latest[group][0]:
                latest[group] = (version, row, mirrored)

    for base, archived in sources:
        for p in sorted(base.rglob("*.json")) if base.is_dir() else []:
            found = _legacy_breakdown_artifact(p, p.relative_to(base).parts)
            if found is not None and archived:
                found = (found[0], found[1], replace(found[2], archived=True))
            offer(found)
    try:
        moves = json.loads((root / "catalogue" / "migration_day_zero.json").read_text())
    except (OSError, ValueError):
        moves = {}
    prefix = LEGACY_BREAKDOWN_PREFIX
    for e in moves.get("entries", []):
        if e["old"].startswith(prefix) and lc.mirrored(e["new"]) and (root / e["new"]).is_file():
            # The current file under its pre-move name: only to know the group's newest.
            rel = tuple(e["old"][len(prefix) :].split("/"))
            offer(_legacy_breakdown_artifact(root / e["new"], rel), mirrored=True)
    return [row for _v, row, mirrored in latest.values() if not mirrored]


def _scan_runtime_cells(results_root: Path = RESULTS_ROOT) -> list[RuntimeCell]:
    """Every sweep ``comparison.json``: mirrored cells and the misc ones (no v3 cell)."""
    cells = []
    for p in sorted(results_root.glob("*/*/*/comparison.json")):
        if p.parts[-4] in lc.DATASET_TYPES:
            cells.append(RuntimeCell(cell=p.parent.parts[-3:], path=p))
    misc = results_root / "misc" / "likelihood_runtime"
    for p in sorted(misc.rglob("comparison.json")) if misc.is_dir() else []:
        cells.append(RuntimeCell(cell=p.parent.relative_to(misc).parts, path=p))
    return sorted(cells, key=lambda c: c.cell)


def _cell_map(root: Path = REPO_ROOT) -> dict[str, str]:
    """Pre-move runtime cell -> its current cell id (``catalogue/series_id_map.json``)."""
    try:
        return json.loads((root / SERIES_ID_MAP).read_text()).get("cells", {})
    except (OSError, ValueError):
        return {}


def _scan_baselines(archive_root: Path = ARCHIVE_ROOT) -> dict[str, list[RuntimeCell]]:
    """Archived named baselines: ``archive/YYYY/MM/<name>/<pre-move cell>/comparison.json``.

    Each cell is re-keyed onto the current cell it measured (the committed cell map); a
    baseline cell the map does not know keeps its pre-move id.
    """
    cell_map = _cell_map(archive_root.parent)
    out: dict[str, list[RuntimeCell]] = {}
    for p in sorted(archive_root.glob("*/*/*/*/*/*/comparison.json")):
        legacy = "/".join(p.parent.parts[-3:])
        cell = tuple(cell_map.get(legacy, legacy).split("/"))
        out.setdefault(p.parts[-5], []).append(RuntimeCell(cell=cell, path=p))
    return out


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


def _config_legacy_seconds(cfg: dict) -> float | None:
    """The legacy per-call full-pipeline value of one comparison.json config entry."""
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


#: Labels of a runtime-table cell whose headline is not the legacy value (timing-noise audit
#: phase 10: the dashboard's estimator rule, ``build_dashboard.headline_reading`` /
#: ``repeat_summary``).
MEDIAN_LABEL = "median"
HEADLINE_FOOTNOTE = (
    "_(median)_: the steady median (>= 5 warm calls, median of individually timed calls) the "
    "row records beside its legacy single-jit value (a block mean in most cells), the same "
    "headline the dashboard uses; "
    "_(median of N runs)_: the median of N independent runs' headlines. Unlabelled values are "
    "the legacy headline (the single-jit block mean, or the step-sum where no single-jit value "
    "exists)."
)


def _config_headline(cfg: dict) -> tuple[float | None, str]:
    """``(seconds, label)`` of one comparison.json config entry: the dashboard's estimator rule.

    The steady median where the row records one beside the single-jit key its headline is
    (labelled ``median``), the median of >= 2 independent runs where the row carries a repeat
    summary (``median of N runs``), else the legacy value (no label).
    """
    legacy = _config_legacy_seconds(cfg)
    if legacy is None:
        return None, ""
    median = headline_median_seconds(cfg)
    seconds, labels = (median, [MEDIAN_LABEL]) if median is not None else (legacy, [])
    estimator = ESTIMATOR_MEDIAN if median is not None else ESTIMATOR_LEGACY
    repeats = repeat_summary(cfg)
    if repeats is not None and repeats["estimator"] == estimator:
        seconds = repeats["median_s"]
        labels.append(f"median of {repeats['n']} runs")
    return seconds, ", ".join(labels)


def _config_headline_seconds(cfg: dict) -> float | None:
    """Per-call full-pipeline headline of one comparison.json config entry (see above)."""
    return _config_headline(cfg)[0]


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

    When archived named baselines exist (``archive/YYYY/MM/<name>/``), one extra
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
    labelled = False

    def _headline_any_config(cell: RuntimeCell) -> tuple[float | None, str]:
        cfgs = cell.configs
        for cname in reversed(_ordered_config_names([cell])):  # prefer A100/extras
            v, label = _config_headline(cfgs.get(cname, {}))
            if v is not None:
                return v, label
        return None, ""

    for cell in cells:
        cfgs = cell.configs
        line = [f"`{cell.cell_id}`"]
        for cname in config_names:
            cfg = cfgs.get(cname, {})
            if cfg.get("cpu_unusable"):
                # Campaign policy (#56) read a run that cannot finish inside the
                # wall-clock cap as GPU-only. Timing-noise audit P9: one such run
                # is one observation, so the label needs a qualified marker
                # (build_dashboard.marker_verdict); otherwise it is inconclusive.
                verdict = marker_verdict(cfg, cname, _RELEASE_SWEEP_CONF)
                line.append(
                    f"**{verdict['label']}**" if verdict["gpu_only"] else f"_{verdict['label']}_"
                )
                continue
            seconds, label = _config_headline(cfg)
            cell_text = _format_time(seconds)
            if label:
                cell_text += f" _({label})_"
                labelled = True
            if seconds is not None and seconds > 60 and not cname.startswith("hpc"):
                # Per-call > 1 min on a local backend: samplers need 1e4-1e6
                # evaluations, so the config is unusable in practice.
                cell_text += " (unusable)"
            line.append(cell_text)
        for bname in baseline_names:
            bcell = baseline_by_cell[bname].get(cell.cell)
            if bcell is None:
                line.append("—")
                continue
            seconds, label = _headline_any_config(bcell)
            line.append(_format_time(seconds) + (f" _({label})_" if label else ""))
            labelled = labelled or bool(label)
        rows.append("| " + " | ".join(line) + " |")
    if labelled:
        rows.append("")
        rows.append(HEADLINE_FOOTNOTE)
    return "\n" + "\n".join(rows) + "\n"


def _inversion_path_label(data: dict, sparse: bool) -> str:
    """Inversion-path label from the payload's ``configuration.inversion_path``
    when present (``sparse_numba`` = the numba CPU cells; the library-dispatch
    interferometer cells record the inversion class name the factory built,
    autolens_profiling#326), else the filename ``_sparse``-tag convention."""
    payload_path = (data.get("configuration") or {}).get("inversion_path")
    if payload_path in ("sparse_numba", "InversionInterferometerSparseNumba"):
        return "sparse (numba)"
    if payload_path == "InversionInterferometerSparse":
        return "sparse (w-tilde)"
    if payload_path == "sparse":
        return "sparse (w-tilde)"
    if payload_path == "dense":
        return "dense (mapping)"
    return "sparse (w-tilde)" if sparse else "dense (mapping)"


def _render_breakdown_table(rows: list[BreakdownRow]) -> str:
    """One row per current breakdown cell (v3 identity) with the step-sum total."""
    if not rows:
        return _no_data_block("run a script under `likelihood_breakdown/` to populate.")
    host_rank = {h: i for i, h in enumerate(HOST_ORDER)}
    out = [
        "| Cell | Instrument | Setup | Host | Inversion path | Step-sum total | PyAutoLens version |",
        "|------|------------|-------|------|----------------|----------------|--------------------|",
    ]
    for row in sorted(
        rows,
        key=lambda r: (
            r.cell,
            r.instrument or "",
            host_rank.get(r.host, len(HOST_ORDER)),
            r.setup or "",
            r.sparse,
        ),
    ):
        data = row.data
        total = data.get("total_step_by_step")
        setup = f"`{row.setup}`" if row.setup else "_not recorded_"
        out.append(
            f"| `{row.cell}` | {row.instrument or '—'} | {setup} | {row.host} | "
            f"{_inversion_path_label(data, row.sparse)} | "
            f"{_format_time(total if isinstance(total, (int, float)) else None)} | "
            f"{row.version_label} |"
        )
    return "\n" + "\n".join(out) + "\n"


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
        rows.append(f"| `{script}.py` | {_format_time(total)} | {art.version_label} |")
    return "\n" + "\n".join(rows) + "\n"


def _render_pipeline_resume_table(artifacts: list[Artifact]) -> str:
    relevant = [a for a in artifacts if a.section == "pipeline_resume"]
    if not relevant:
        return _no_data_block(
            "run `pipeline_resume/slam_resume.py` twice (cold, then resume) to "
            "populate. See section README."
        )
    latest = _latest_per_group(relevant, key=lambda a: (a.script, a.instrument))
    rows = [
        "| Script | Instrument | Cold total | Resume total | Imports | "
        "Σ stage resume | Σ inter-stage | PyAutoLens version |"
    ]
    rows.append(
        "|--------|------------|------------|--------------|---------|"
        "----------------|---------------|--------------------|"
    )

    def _sum_spans(run: dict, component: str) -> float:
        return sum(v for k, v in run.get("spans", {}).items() if k.endswith(f"/{component}"))

    for (script, instrument), art in sorted(latest.items()):
        runs = art.data.get("runs", [])
        cold = next((r for r in runs if r.get("mode") == "cold"), None)
        resume = next((r for r in reversed(runs) if r.get("mode") == "resume"), None)
        cold_total = _format_time(cold.get("total_s")) if cold else "—"
        if resume:
            resume_total = _format_time(resume.get("total_s"))
            imports = _format_time(resume.get("import_s"))
            stage = _format_time(_sum_spans(resume, "search_fit"))
            inter = _format_time(
                sum(_sum_spans(resume, c) for c in ("adapt_images", "positions", "model_compose"))
            )
        else:
            resume_total = imports = stage = inter = "—"
        rows.append(
            f"| `{script}.py` | {instrument or '—'} | {cold_total} | "
            f"{resume_total} | {imports} | {stage} | {inter} | {art.version_label} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _render_headline(
    breakdown: list[BreakdownRow],
    cells: list[RuntimeCell],
    baselines: dict[str, list[RuntimeCell]],
) -> str:
    """Top-level dashboard: runtime matrix + latest breakdown totals."""
    parts = ["\n**Likelihood runtime** — full-pipeline per-call cost per cell × config:\n"]
    parts.append(_render_runtime_table(cells, baselines).strip("\n"))
    parts.append("\n**Likelihood breakdown** — latest per-step decompositions:\n")
    parts.append(_render_breakdown_table(breakdown).strip("\n"))
    return "\n" + "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Region registry + rewrite driver
# ---------------------------------------------------------------------------


def _render_jax_compile_warm_table() -> str:
    """Pinned WARM compile per cell/transform, grouped by comparability key.

    Grouped, never merged into one table: a warm compile is only comparable
    within `(hardware, jax_version, mixed_precision, cache_state)`, so a single
    ranked table across keys would invite exactly the cross-key comparison the
    pins exist to prevent.
    """
    import json as _json

    pins_path = _MISC / "jax_compile" / "pins.json"
    if not pins_path.is_file():
        return "\n_No compile pins yet — run `jax_compile/update_pins.py --write`._\n"
    try:
        pins = _json.loads(pins_path.read_text()).get("pins") or []
    except (OSError, ValueError):
        return "\n_Compile pins unreadable._\n"
    if not pins:
        return "\n_No warm compile records pinned yet._\n"

    groups: dict[tuple, list[dict]] = {}
    for pin in pins:
        key = (
            pin.get("hardware"),
            pin.get("hostname"),
            pin.get("jax_version"),
            bool(pin.get("mixed_precision")),
        )
        groups.setdefault(key, []).append(pin)

    out: list[str] = []
    for (hardware, host, jax_version, mp), rows in sorted(
        groups.items(), key=lambda kv: str(kv[0])
    ):
        out.append(
            f"**`{hardware}` · `{host}` · jax {jax_version}{' · mixed-precision' if mp else ''}**"
        )
        out.append("")
        out.append("| Cell | Transform | Warm compile | Source |")
        out.append("|---|---|---|---|")
        for pin in sorted(
            rows,
            key=lambda p: (
                str(p.get("dataset_class")),
                str(p.get("model_type")),
                str(p.get("instrument")),
                str(p.get("transform")),
            ),
        ):
            cell = "/".join(
                str(pin.get(f)) for f in ("dataset_class", "model_type", "instrument") if pin.get(f)
            )
            out.append(
                f"| `{cell}` | `{pin.get('transform')}` | "
                f"{_format_time(pin.get('compile_s'))} | "
                f"`{pin.get('source_tag') or '—'}` {pin.get('source_timestamp') or ''} |"
            )
        out.append("")
    return "\n" + "\n".join(out).rstrip() + "\n"


def _render_hazards_table() -> str:
    """Committed numerical findings, keyed by stable semantic ID."""

    index_path = RESULTS_ROOT / "misc" / "hazards" / "hazards_index.json"
    if not index_path.is_file():
        return "\n_No hazard findings yet — run `hazards/scan.py`._\n"
    try:
        findings = json.loads(index_path.read_text()).get("findings") or {}
    except (OSError, ValueError):
        return "\n_Hazard index unreadable._\n"
    if not findings:
        return "\n_No hazard findings yet — run `hazards/scan.py`._\n"

    rows = [
        "| Finding | Subject | Hazard | Risk basis | Backends |",
        "|---|---|---|---|---|",
    ]
    for finding_id, finding in sorted(findings.items()):
        bases = ", ".join(
            sorted(
                {
                    measurement.get("basis", "")
                    for measurement in finding.get("measurements", [])
                    if measurement.get("basis")
                }
            )
        )
        rows.append(
            f"| `{finding_id}` | `{finding.get('subject')}` | "
            f"`{finding.get('hazard_class')}` | {bases or '—'} | "
            f"{', '.join(finding.get('backends', [])) or '—'} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _deflection_pin_label(data: dict, name: str) -> str:
    """Pin status for one profile row of a deflection artifact.

    ``new`` — the run created the pins (first run for that instrument, or a
    deliberate ``--repin``), so nothing was verified on it; ``ok`` — every
    pinned value matched at rtol 1e-6; ``DRIFT`` — at least one did not, so the
    row's timings are not comparable to the pinned baseline.
    """
    if data.get("pin_provenance"):
        return "new"
    expected = data.get("pinned_expected") or {}
    if name not in expected:
        return "—"
    prefix = f"{name}."
    for record in data.get("pinned_drift") or []:
        if str(record.get("label", "")).startswith(prefix):
            return "**DRIFT**"
    return "ok"


def _render_deflections_table(artifacts: list[Artifact]) -> str:
    """One row per (cell, instrument, mass profile) with the three per-call times."""
    relevant = [a for a in artifacts if a.section == "lens" and a.subfolder == "deflections"]
    if not relevant:
        return _no_data_block(
            "run a cell under `scripts/lens/deflections/` to populate. See section README."
        )
    latest = _latest_per_group(relevant, key=lambda a: (a.script, a.instrument))
    rows = [
        "| Cell | Instrument | Profile | Grid2D s/call | Irregular s/call | "
        "Tracer s/call | Pin | Version |",
        "|------|------------|---------|---------------|------------------|"
        "---------------|-----|---------|",
    ]
    for (script, instrument), art in sorted(
        latest.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
    ):
        data = art.data
        for name, record in (data.get("profiles") or {}).items():
            rows.append(
                f"| `{script}` | "
                f"{instrument or '—'} | "
                f"`{name}` | "
                f"{_format_time(record.get('grid2d_s'))} | "
                f"{_format_time(record.get('irregular_s'))} | "
                f"{_format_time(record.get('tracer_s'))} | "
                f"{_deflection_pin_label(data, name)} | "
                f"v{art.raw_version} |"
            )
    return "\n" + "\n".join(rows) + "\n"


def _sci(value) -> str:
    """Compact scientific formatting for the solver tables (``—`` for missing)."""
    if value is None or not isinstance(value, (int, float)) or math.isnan(value):
        return "—"
    return f"{value:.2e}"


def _render_solver_corpus_table() -> str:
    """One row per corpus group of ``results/lens/solver/corpus/manifest.json``."""
    manifest_path = RESULTS_ROOT / "lens" / "solver" / "corpus" / "manifest.json"
    if not manifest_path.is_file():
        return _no_data_block("run `python scripts/lens/solver/_corpus.py` to build the corpus.")
    try:
        groups = json.loads(manifest_path.read_text()).get("groups") or []
    except (OSError, ValueError):
        return "\n_Corpus manifest unreadable._\n"
    if not groups:
        return _no_data_block("the corpus manifest has no groups yet.")
    rows = [
        "| Group | Systems | n | cond(Q) | max abs(q) | Source columns | Captured by | Model |",
        "|-------|---------|---|---------|------------|----------------|-------------|-------|",
    ]
    for group in groups:
        systems = group.get("systems") or []
        ns = sorted({s.get("n") for s in systems if s.get("n") is not None})
        conds = [s["cond_Q"] for s in systems if s.get("cond_Q") is not None]
        qs = [s["max_abs_q"] for s in systems if s.get("max_abs_q") is not None]
        n_src = sorted({s.get("n_source_columns") for s in systems}, key=lambda v: (v is None, v))
        models = sorted({s.get("model") or "—" for s in systems})
        script = (group.get("source") or {}).get("script") or "—"
        rows.append(
            f"| `{group.get('name')}` | {len(systems)} | "
            f"{', '.join(str(n) for n in ns) or '—'} | "
            f"{_sci(min(conds)) if conds else '—'} – {_sci(max(conds)) if conds else '—'} | "
            f"{_sci(min(qs)) if qs else '—'} – {_sci(max(qs)) if qs else '—'} | "
            f"{', '.join('—' if v is None else str(v) for v in n_src)} | "
            f"`{script}` | {'; '.join(models)} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _solver_latest(artifacts: list[Artifact], script: str) -> dict[tuple, Artifact]:
    relevant = [
        a
        for a in artifacts
        if a.section == "lens" and a.subfolder == "solver" and a.script == script
    ]
    return _latest_per_group(relevant, key=lambda a: (a.instrument,))


def _render_solver_accuracy_table(artifacts: list[Artifact]) -> str:
    """Per-candidate aggregates of the latest ``accuracy`` artifact per corpus label."""
    latest = _solver_latest(artifacts, "accuracy")
    if not latest:
        return _no_data_block("run `python scripts/lens/solver/accuracy.py` to populate.")
    rows = [
        "| Corpus | Candidate | Unconverged | Non-finite | Worst amp_rel_max | "
        "Worst amp_rel_max (sig) | Worst abs(flux_rel_source) | Worst abs(flux_inactive_rel) | "
        "Worst objective gap | Worst KKT | Median iters | Median wall ms | Version |",
        "|--------|-----------|-------------|------------|-------------------|"
        "-------------------------|----------------------------|------------------------------|"
        "---------------------|-----------|--------------|----------------|---------|",
    ]
    for (label,), art in sorted(latest.items(), key=lambda kv: kv[0][0] or ""):
        for name, agg in (art.data.get("aggregates") or {}).items():
            n = agg.get("n_systems")
            wall = agg.get("median_wall_ms")
            rows.append(
                f"| `{label or '—'}` | `{name}` | {agg.get('n_unconverged')}/{n} | "
                f"{agg.get('n_nonfinite')} | {_sci(agg.get('worst_amp_rel_max'))} | "
                f"{_sci(agg.get('worst_amp_rel_max_sig'))} | "
                f"{_sci(agg.get('worst_flux_rel_source'))} | "
                f"{_sci(agg.get('worst_flux_inactive_rel'))} | "
                f"{_sci(agg.get('worst_objective_gap'))} | "
                f"{_sci(agg.get('worst_kkt_residual_scaled'))} | "
                f"{agg.get('median_iterations') if agg.get('median_iterations') is not None else '—'} | "
                f"{'—' if wall is None else f'{wall:.3f}'} | v{art.raw_version} |"
            )
    return "\n" + "\n".join(rows) + "\n"


def _render_solver_early_stopping_table(artifacts: list[Artifact]) -> str:
    """Per-cap aggregates of the latest ``early_stopping`` artifact per corpus label."""
    latest = _solver_latest(artifacts, "early_stopping")
    if not latest:
        return _no_data_block("run `python scripts/lens/solver/early_stopping.py` to populate.")
    rows = [
        "| Corpus | Cap | Converged | Worst amp_rel_max | Worst amp_rel_max (sig) | "
        "Median amp_rel_max | Worst abs(flux_rel_source) | Worst abs(flux_inactive_rel) | "
        "Worst KKT | Median wall ms | Version |",
        "|--------|-----|-----------|-------------------|-------------------------|"
        "--------------------|----------------------------|------------------------------|"
        "-----------|----------------|---------|",
    ]
    for (label,), art in sorted(latest.items(), key=lambda kv: kv[0][0] or ""):
        aggs = art.data.get("aggregates") or {}
        for cap in sorted(aggs, key=lambda c: int(c)):
            agg = aggs[cap]
            n = agg.get("n_systems") or 0
            wall = agg.get("median_wall_ms")
            rows.append(
                f"| `{label or '—'}` | {cap} | {n - (agg.get('n_unconverged') or 0)}/{n} | "
                f"{_sci(agg.get('worst_amp_rel_max'))} | "
                f"{_sci(agg.get('worst_amp_rel_max_sig'))} | "
                f"{_sci(agg.get('median_amp_rel_max'))} | "
                f"{_sci(agg.get('worst_flux_rel_source'))} | "
                f"{_sci(agg.get('worst_flux_inactive_rel'))} | "
                f"{_sci(agg.get('worst_kkt_residual_scaled'))} | "
                f"{'—' if wall is None else f'{wall:.3f}'} | v{art.raw_version} |"
            )
    return "\n" + "\n".join(rows) + "\n"


def _render_solver_timing_table(artifacts: list[Artifact]) -> str:
    """Per-(candidate, batch family, batch size) rows of the latest ``timing`` artifact per label.

    Steady per-evaluation cost (min / median over interleaved rounds, wall / B) and the
    first-call compile wall are separate columns; a compile wall whose shape was already
    compiled by an earlier family is marked ``(cached)``.
    """
    latest = _solver_latest(artifacts, "timing")
    if not latest:
        return _no_data_block("run `python scripts/lens/solver/timing.py` to populate.")
    rows = [
        "| Corpus | Candidate | Batch family | B | Per-eval min ms | Per-eval median ms | "
        "Batch wall min ms | Compile s | Median / max iters | Unconverged | "
        "Worst abs(flux_inactive_rel) | Version |",
        "|--------|-----------|--------------|---|-----------------|--------------------|"
        "-------------------|-----------|--------------------|-------------|"
        "------------------------------|---------|",
    ]
    for (label,), art in sorted(latest.items(), key=lambda kv: kv[0][0] or ""):
        for r in art.data.get("rows") or []:
            fam = r.get("batch_family") or "—"
            if r.get("tiled"):
                fam += " (tiled)"
            if r.get("looped"):
                fam += " (host loop)"
            compile_s = r.get("compile_s")
            compile_txt = "—" if compile_s is None else f"{compile_s:.2f}"
            if r.get("compiled_here") is False:
                compile_txt += " (cached)"
            med = r.get("median_iterations")
            rows.append(
                f"| `{label or '—'}` | `{r.get('candidate')}` | {fam} | {r.get('batch_size')} | "
                f"{r.get('per_eval_min_ms', float('nan')):.4f} | "
                f"{r.get('per_eval_median_ms', float('nan')):.4f} | "
                f"{r.get('wall_min_ms', float('nan')):.3f} | {compile_txt} | "
                f"{'—' if med is None else f'{med:g}'} / {r.get('max_iterations')} | "
                f"{r.get('n_unconverged')}/{r.get('batch_size')} | "
                f"{_sci(r.get('worst_flux_inactive_rel'))} | v{art.raw_version} |"
            )
    return "\n" + "\n".join(rows) + "\n"


def _render_solver_accuracy_posthoc_table(artifacts: list[Artifact]) -> str:
    """Per-candidate aggregates of the latest ``accuracy_posthoc`` artifact per corpus label.

    POST-HOC (exploratory, not pre-registered) candidates, kept out of the pre-registered
    ``solver-accuracy`` table on purpose.
    """
    latest = _solver_latest(artifacts, "accuracy_posthoc")
    if not latest:
        return _no_data_block("run `python scripts/lens/solver/accuracy.py --posthoc` to populate.")
    rows = [
        "| Corpus | Candidate | Unconverged | Worst abs(flux_inactive_rel) | "
        "Worst abs(flux_rel_source) | Worst amp_rel_max (sig) | Max active-set mismatch | "
        "Median iters | Max iters | Median wall ms | Version |",
        "|--------|-----------|-------------|------------------------------|"
        "----------------------------|-------------------------|-------------------------|"
        "--------------|-----------|----------------|---------|",
    ]
    for (label,), art in sorted(latest.items(), key=lambda kv: kv[0][0] or ""):
        for name, agg in (art.data.get("aggregates") or {}).items():
            n = agg.get("n_systems")
            wall = agg.get("median_wall_ms")
            mism = agg.get("worst_active_set_mismatch")
            rows.append(
                f"| `{label or '—'}` | `{name}` | {agg.get('n_unconverged')}/{n} | "
                f"{_sci(agg.get('worst_flux_inactive_rel'))} | "
                f"{_sci(agg.get('worst_flux_rel_source'))} | "
                f"{_sci(agg.get('worst_amp_rel_max_sig'))} | "
                f"{'—' if mism is None else mism} | "
                f"{agg.get('median_iterations') if agg.get('median_iterations') is not None else '—'} | "
                f"{agg.get('max_iterations') if agg.get('max_iterations') is not None else '—'} | "
                f"{'—' if wall is None else f'{wall:.3f}'} | v{art.raw_version} |"
            )
    return "\n" + "\n".join(rows) + "\n"


def _render_solver_euclid_latent_table() -> str:
    """The latest ``euclid_latent_by_candidate_v<version>.json``: candidate -> latent."""
    folder = RESULTS_ROOT / "lens" / "solver"
    found = []
    for p in folder.glob("euclid_latent_by_candidate_v*.json"):
        raw = p.stem.rsplit("_v", 1)[1]
        try:
            found.append((_parse_version(raw), raw, p))
        except ValueError:
            continue
    if not found:
        return _no_data_block("run `python scripts/lens/solver/euclid_latent.py` to populate.")
    _, raw, path = max(found)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return "\n_euclid latent JSON unreadable._\n"
    ref = data.get("latent_reference") or {}
    val = data.get("validation") or {}
    rtol = data.get("test_rtol")
    rows = [
        f"_System `{data.get('system')}`; eager NumPy `total_source_flux` "
        f"{ref.get('eager_numpy')}, released jit {ref.get('jit_released')}"
        + (
            f", library-main jit {ref['jit_library_main']}"
            if ref.get("jit_library_main") is not None
            else ""
        )
        + "; validation "
        f"{'passed' if val.get('passed') else 'FAILED'} (x_ref rel "
        f"{_sci((val.get('x_ref_to_eager') or {}).get('rel'))}, pdip_raw rel "
        f"{_sci((val.get('pdip_raw_to_jit') or {}).get('rel'))}); the euclid test's rtol is "
        f"{rtol}._",
        "",
        "| Candidate | Post-hoc | total_source_flux | rel vs eager | Within test rtol | "
        "Converged | Iterations | Version |",
        "|-----------|----------|-------------------|--------------|------------------|"
        "-----------|------------|---------|",
    ]
    for r in data.get("rows") or []:
        flux = r.get("total_source_flux")
        rel = r.get("rel_vs_eager")
        ok = r.get("passes_test_rtol")
        rows.append(
            f"| `{r.get('candidate')}` | {'yes' if r.get('posthoc') else 'no'} | "
            f"{'—' if flux is None else f'{flux:.7f}'} | "
            f"{'—' if rel is None else f'{rel:+.2e}'} | "
            f"{'—' if ok is None else ('yes' if ok else 'no')} | {r.get('converged')} | "
            f"{r.get('iterations')} | v{raw} |"
        )
    return "\n" + "\n".join(rows) + "\n"


def _build_renderers():
    artifacts = _scan_artifacts()
    breakdown = _scan_breakdown_rows()
    cells = _scan_runtime_cells()
    baselines = _scan_baselines()
    return artifacts, {
        "headline": lambda: _render_headline(breakdown, cells, baselines),
        "runtime": lambda: _render_runtime_table(cells, baselines),
        "breakdown": lambda: _render_breakdown_table(breakdown),
        "simulators": lambda: _render_simulator_table(artifacts),
        "pipeline-resume": lambda: _render_pipeline_resume_table(artifacts),
        "jax-compile-warm": _render_jax_compile_warm_table,
        "hazards": _render_hazards_table,
        "deflections": lambda: _render_deflections_table(artifacts),
        "solver-corpus": _render_solver_corpus_table,
        "solver-accuracy": lambda: _render_solver_accuracy_table(artifacts),
        "solver-early-stopping": lambda: _render_solver_early_stopping_table(artifacts),
        "solver-accuracy-posthoc": lambda: _render_solver_accuracy_posthoc_table(artifacts),
        "solver-euclid-latent": _render_solver_euclid_latent_table,
        "solver-timing": lambda: _render_solver_timing_table(artifacts),
    }


# Files that may contain auto-table regions. Listing them explicitly (rather
# than walking the repo) keeps the script's surface obvious.
# After the dataset-first restructure, per-task shared material (drivers +
# narrative READMEs with the auto-tables) lives under scripts/misc/<task>/.
_MISC = REPO_ROOT / "scripts" / "misc"
TARGET_READMES = [
    REPO_ROOT / "README.md",
    _MISC / "likelihood_runtime" / "README.md",
    _MISC / "likelihood_breakdown" / "README.md",
    _MISC / "simulators" / "README.md",
    _MISC / "pipeline_resume" / "README.md",
    _MISC / "jax_compile" / "README.md",
    _MISC / "hazards" / "README.md",
    REPO_ROOT / "scripts" / "lens" / "deflections" / "README.md",
    REPO_ROOT / "scripts" / "lens" / "solver" / "README.md",
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
