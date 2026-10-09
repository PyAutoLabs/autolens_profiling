"""Build the setup browser and retained historical feeds: ``results/**`` -> ``dashboard/``.

Run from the repo root::

    python scripts/misc/tooling/build_dashboard.py           # write dashboard/{series,state}.json + index.html
    python scripts/misc/tooling/build_dashboard.py --check   # exit 1 if the committed pages are stale

Historical transport retained during browser migration
-----------------------------------------------------

One point per PyAutoLens release per **series** — a likelihood cell x instrument x sweep
config (device + precision) x dense/sparse — read from the result JSONs this repo already
commits (autolens_profiling#345):

* versioned artifacts ``<script>_<summary|breakdown>_<instrument>_v<version>[_sparse].json``
  under ``results/{runtime,breakdown,simulators,lens}`` (untagged = ``local_cpu_fp64`` by
  package convention, the same grammar as ``build_readme.py``);
* config-tagged rows ``<script>_<config>[_sparse].json`` under ``results/{runtime,breakdown}``
  whose payload carries ``autolens_version`` (the A100 and RAL CPU tiers);
* every sweep ``comparison.json`` under ``results/runtime`` (one entry per config).

The per-call headline is the ladder ``build_readme.py`` uses (``full_pipeline_per_call`` ->
``full_pipeline_single_jit`` -> ...), extended for breakdown cells; ``vmap.per_call`` rides
beside it. A point is **qualified** when it is on a reference host class (``hpc_*``; laptop rows
never qualify as trend points) and its provenance block (``device.provenance``,
autolens_profiling#342) shows a host and a load average, the pinned reference host, under the
load-average cap in ``hpc/release_sweep.conf``; a row above the cap is **refused** (listed, not
plotted); every other unqualified row is plotted hollow with its reason.

The drift badge compares a series' last two releases with the profiling conductor's ratio
(``>= 2.0x``, ``PyAutoBrain/agents/conductors/profiling``); the absolute floor here is 1 ms
per call, this dashboard's own choice (the conductor's 1 s floor is for compile times). The
badge is a flag for the conductor's ``triage``, never a verdict.

Outputs
-------

``dashboard/series.json`` (the trend data), ``dashboard/state.json`` (the project badge feed,
``PyAutoBrain/board/state_schema.json`` v1), ``dashboard/summary.json`` (the ``profiling-summary``
v1 read contract the PyAutoPulse organ ingests -- ``dashboard/README.md``) and
``dashboard/index.html`` (setup browser when a catalogue is registered, historical page otherwise).
The browser uses Brain's shared theme and the versioned setup catalogue; no scientific imports.
``lint.yml`` runs ``--check``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
PAGES_URL = "https://pyautolabs.github.io/autolens_profiling/"
REPO_URL = "https://github.com/PyAutoLabs/autolens_profiling"

#: The profiling conductor's drift ratio (COMPILE_DRIFT_RATIO in
#: PyAutoBrain/agents/conductors/profiling/_profiling.py), applied to per-call run time.
DRIFT_RATIO = 2.0
#: Absolute floor for run-time drift, seconds per call: sub-millisecond wobble on a 0.4 ms
#: cell is not a 2x regression worth a badge.
DRIFT_FLOOR_S = 0.001

#: Config-label prefixes measured on a declared reference host class. Only these rows can be
#: trend points: laptop ``local_*`` rows never qualify, provenance or not
#: (``hpc/release_sweep.conf``: "the laptop drifts 2.5x between runs"). One rule, reused by any
#: future ``profiling-summary`` v2 / ``catalogue.json`` qualification (timing-noise audit P6).
REFERENCE_HOST_CLASS_PREFIXES = ("hpc_",)

#: The point field that would carry a repeat summary of the compared metric (the number of
#: independent repeats -- separate runs -- behind ``single_jit_s``). No producer writes it: a
#: headline is one 10-call block mean or, since fix phase 5 (timing-noise audit P8), one steady
#: median of one process's individually timed calls. Either is one summary from one run, so every
#: endpoint is still single-sample; the timed calls inside one run are not independent repeats.
#: ``_point`` does not copy it from a result JSON: a producer that runs >= 2 independent repeats per
#: release must add it there too. Until then this fails safe.
REPEAT_SUMMARY_FIELD = "single_jit_repeats"
#: Reasons the comparisons carry for the 2x band (timing-noise audit P7).
SINGLE_SAMPLE_NULL_REASON = (
    "single-sample endpoint(s): within the 2x policy band is not a measured null"
)
FLAT_BAND_REASON = "within the 2x policy band; not a measured null"
SINGLE_SAMPLE_REASON = "single-sample endpoint(s)"

#: Headline estimators (timing-noise audit P8). A point's ``single_jit_s`` is the steady median
#: (``full_pipeline_single_jit_median_ms``: >= 5 warm calls, median of individually timed calls)
#: where its row records one beside a ``full_pipeline_single_jit`` headline, else the legacy
#: ``HEADLINE_KEYS`` value. Points carry ``headline_estimator`` only when it is the median, so a
#: missing field means the legacy headline. Drift compares like with like only.
ESTIMATOR_MEDIAN = "steady median"
ESTIMATOR_LEGACY = "legacy headline"
ESTIMATOR_MISMATCH_REASON = (
    "endpoints use different headline estimators (steady median vs legacy block mean): not compared"
)

#: How a CPU ``.unusable.json`` marker renders (timing-noise audit P9). ``GPU-only`` only when the
#: marker itself qualifies under ``qualify`` (reference host class, host and load recorded, load
#: under the cap, the pinned node); every other marker is inconclusive and is re-measured by
#: ``sweep.py --skip-existing``.
MARKER_GPU_ONLY = "GPU-only"
MARKER_TIMED_OUT = "timed out (inconclusive)"
MARKER_NOT_FINISHED = "did not finish (inconclusive)"

#: The ``profiling-summary`` read contract this project publishes for the PyAutoPulse organ
#: (``dashboard/README.md``; design: PyAutoBrain/docs/research/profiling_inference_organs.md).
#: The envelope and record grammar are the exchange contract the organ validates; the series
#: semantics, qualification and drift policy above stay this project's and are only described.
SUMMARY_SCHEMA = "profiling-summary"
SUMMARY_VERSION = 1
SUMMARY_PROJECT = "autolens_profiling"
SUMMARY_SCOPE = "release-runtime"
#: Identifier of the drift verdict the comparisons carry -- the 2x ratio + 1 ms floor above.
DRIFT_POLICY = "runtime-drift-2x-1ms"
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Same filename grammar as build_readme.py (kept in step by test_build_dashboard.py).
ARTIFACT_RE = re.compile(
    r"^(?P<script>[a-z0-9_]+?)_(?P<purpose>summary|breakdown)"
    r"(?:_(?P<extra>[a-z0-9_]+?))?"
    r"_v(?P<version>[0-9]+(?:\.[0-9]+)+)"
    r"(?P<sparse>_sparse)?"
    r"\.json$"
)
CONFIG_TAGGED_RE = re.compile(
    r"^(?P<script>[a-z0-9_]+?)_(?P<config>local_cpu_fp64|local_cpu_mp|"
    r"local_gpu_fp64|local_gpu_mp|hpc_ral_cpu_fp64|hpc_a100_fp64|hpc_a100_mp)"
    r"(?P<sparse>_sparse)?"
    r"\.json$"
)
CONFIG_ORDER = (
    "local_cpu_fp64",
    "local_cpu_mp",
    "local_gpu_fp64",
    "local_gpu_mp",
    "hpc_ral_cpu_fp64",
    "hpc_a100_fp64",
    "hpc_a100_mp",
)
SECTIONS = ("runtime", "breakdown", "simulators", "lens")
HEADLINE_KEYS = (
    "full_pipeline_per_call",
    "full_pipeline_single_jit",
    "full_pipeline_cube_single_jit",
    "total_step_by_step_cube",
    # breakdown cells: the direct call first, then the step-by-step total
    "direct_log_likelihood_function_per_call",
    "total_step_by_step",
    # simulators / lens summaries
    "total_s",
    "per_call_s",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


def read_release_sweep_conf(root: Path) -> dict:
    """``hpc/release_sweep.conf`` as {node, loadavg_cap}; absent file -> no pin, no cap."""
    conf = {"node": None, "loadavg_cap": None}
    path = root / "hpc" / "release_sweep.conf"
    if not path.is_file():
        return conf
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "RELEASE_SWEEP_NODE" and value:
            conf["node"] = value
        elif key == "RELEASE_SWEEP_LOADAVG_CAP" and value:
            try:
                conf["loadavg_cap"] = float(value)
            except ValueError:
                pass
    return conf


def _version_key(v: str) -> tuple:
    parts = []
    for tok in v.split("."):
        parts.append(int(tok) if tok.isdigit() else -1)
    return tuple(parts)


def _finite(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def headline_key(payload: dict) -> str | None:
    """The ``HEADLINE_KEYS`` entry the per-call headline is read from (None: no headline)."""
    for key in HEADLINE_KEYS:
        if _finite(payload.get(key)) is not None:
            return key
    return None


def headline_seconds(payload: dict) -> float | None:
    key = headline_key(payload)
    return None if key is None else _finite(payload.get(key))


#: The label for a GPU row whose headline is ``full_pipeline_single_jit``. That statistic is the
#: mean of one 10-call block right after the first call, which on the A100 can land in the
#: post-compile transient: the source-plane cell read 0.642 ms against a steady 0.267 ms median
#: (jobs 366912 / 366914; autolens_profiling#371). The value is kept for continuity and labelled,
#: not re-based.
FIRST_BLOCK_NOTE = "first block after compile"


def headline_is_single_jit(payload: dict) -> bool:
    """True when the headline is the ``full_pipeline_single_jit`` statistic.

    That is either the key itself, or the sweep aggregator's ``full_pipeline_per_call`` alias of
    it. ``aggregate.py`` copies ``full_pipeline_single_jit`` into ``full_pipeline_per_call`` in
    every ``comparison.json`` entry.
    """
    key = headline_key(payload)
    if key == "full_pipeline_single_jit":
        return True
    single = _finite(payload.get("full_pipeline_single_jit"))
    return (
        key == "full_pipeline_per_call"
        and single is not None
        and single == headline_seconds(payload)
    )


def single_jit_median_seconds(payload: dict) -> float | None:
    """The steady median written beside ``full_pipeline_single_jit`` (#371), in seconds."""
    v = _finite(payload.get("full_pipeline_single_jit_median_ms"))
    return None if v is None else v / 1000.0


def headline_estimator(point: dict) -> str:
    """The estimator behind a point's ``single_jit_s`` (absent field: the legacy headline)."""
    return point.get("headline_estimator", ESTIMATOR_LEGACY)


def vmap_seconds(payload: dict) -> float | None:
    vmap = payload.get("vmap")
    if isinstance(vmap, dict):
        return _finite(vmap.get("per_call"))
    return None


def _provenance(payload: dict) -> dict:
    """host / job / loadavg / hostname from a payload's device block (all optional)."""
    device = payload.get("device") if isinstance(payload.get("device"), dict) else {}
    prov = device.get("provenance") if isinstance(device.get("provenance"), dict) else None
    out = {
        "hostname": device.get("hostname") if isinstance(device.get("hostname"), str) else None,
        "backend": device.get("backend") if isinstance(device.get("backend"), str) else None,
        "has_provenance": prov is not None,
        "host": None,
        "job": None,
        "loadavg": None,
    }
    if prov:
        out["host"] = prov.get("host") if isinstance(prov.get("host"), str) else None
        slurm = prov.get("slurm") if isinstance(prov.get("slurm"), dict) else {}
        out["job"] = slurm.get("job_id")
        la = prov.get("loadavg_at_import")
        if isinstance(la, list) and la and _finite(la[0]) is not None:
            out["loadavg"] = float(la[0])
    if out["host"] is None and out["hostname"]:
        out["host"] = out["hostname"]
    return out


def warmup_unsettled(payload: dict) -> str | None:
    """Why a payload's recorded warm-up never settled, or None (timing-noise audit P3).

    Only a payload carrying a warm-up *record* (a dict, as ``fixed_light_numba.py`` writes under
    ``rows[*].warmup``) is judged; a scalar warm-up time or no field at all is not a record. The
    rule is the shared ``likelihood_breakdown.warmup_gate``, the one the overhead verdict and the
    promotion block apply.
    """
    warmup = payload.get("warmup")
    if not isinstance(warmup, dict):
        return None
    misc = str(Path(__file__).resolve().parents[1])
    if misc not in sys.path:
        sys.path.insert(0, misc)
    from likelihood_breakdown.warmup_gate import warmup_unsettled_reason

    return warmup_unsettled_reason(warmup)


def _load(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _point(payload: dict, version: str, source: str) -> dict | None:
    legacy = headline_seconds(payload)
    median = single_jit_median_seconds(payload)
    # The median is the headline only beside a single-jit headline it summarises (P8).
    use_median = median is not None and headline_is_single_jit(payload)
    single = median if use_median else legacy
    vmap = vmap_seconds(payload)
    if single is None and vmap is None:
        return None
    prov = _provenance(payload)
    point = {
        "version": version,
        "single_jit_s": single,
        "vmap_per_call_s": vmap,
        "host": prov["host"],
        "backend": prov["backend"],
        "job": prov["job"],
        "loadavg": prov["loadavg"],
        "has_provenance": prov["has_provenance"],
        "source": source,
    }
    # Added only where they apply, so every other point is unchanged.
    if use_median:
        point["headline_estimator"] = ESTIMATOR_MEDIAN
        point["single_jit_block_mean_s"] = legacy
    elif prov["backend"] == "gpu" and headline_is_single_jit(payload):
        point["headline_note"] = FIRST_BLOCK_NOTE
    if median is not None:
        point["single_jit_median_s"] = median
    unsettled = warmup_unsettled(payload)
    if unsettled is not None:
        point["warmup_unsettled"] = unsettled
    return point


def _per_call_html(p: dict) -> str:
    """The table's per-call cell: the headline, which estimator it is, and the other estimator."""
    out = html.escape(_fmt_s(p["single_jit_s"]))
    if headline_estimator(p) == ESTIMATOR_MEDIAN:
        out += f' <span class="muted">({html.escape(ESTIMATOR_MEDIAN)})</span>'
        if p.get("single_jit_block_mean_s") is not None:
            out += (
                f'<br><span class="muted">block mean {html.escape(_fmt_s(p["single_jit_block_mean_s"]))}'
                f" ({html.escape(FIRST_BLOCK_NOTE)})</span>"
            )
        return out
    if p.get("headline_note"):
        out += f' <span class="muted">({html.escape(p["headline_note"])})</span>'
    if p.get("single_jit_median_s") is not None:
        out += f'<br><span class="muted">steady median {html.escape(_fmt_s(p["single_jit_median_s"]))}</span>'
    return out


def scan(root: Path) -> dict[tuple, list[dict]]:
    """Every (section, cell, config, sparse) -> its points, one per (version, source)."""
    root = root.resolve()
    results = root / "results"
    raw: dict[tuple, list[dict]] = {}

    def add(key: tuple, point: dict | None):
        if point is not None:
            raw.setdefault(key, []).append(point)

    for section in SECTIONS:
        base = results / section
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            rel = path.relative_to(root)
            parts = path.relative_to(base).parts
            if path.name == "comparison.json":
                if section != "runtime":
                    continue
                payload = _load(path)
                configs = payload.get("configs") if payload else None
                if not isinstance(configs, dict):
                    continue
                cell = "/".join(parts[:-1])
                for config, entry in configs.items():
                    if not isinstance(entry, dict) or "autolens_version" not in entry:
                        continue
                    name, sparse = (
                        (config[:-7], True) if config.endswith("_sparse") else (config, False)
                    )
                    add(
                        (section, cell, name, sparse),
                        _point(entry, str(entry["autolens_version"]), f"{rel}#{config}"),
                    )
                continue
            m = ARTIFACT_RE.match(path.name)
            if m:
                payload = _load(path)
                if payload is None:
                    continue
                folder = "/".join(parts[:-1])
                instrument = m["extra"] or (
                    payload.get("instrument")
                    if isinstance(payload.get("instrument"), str)
                    else None
                )
                if section == "runtime":
                    cell = (
                        folder
                        if (instrument and folder.endswith("/" + instrument))
                        else (f"{folder}/{instrument}" if instrument else folder)
                    )
                else:
                    cell = f"{folder}/{m['script']}" + (f"/{instrument}" if instrument else "")
                add(
                    (section, cell, "local_cpu_fp64", bool(m["sparse"])),
                    _point(payload, m["version"], str(rel)),
                )
                continue
            m = CONFIG_TAGGED_RE.match(path.name)
            if m and section in ("runtime", "breakdown"):
                payload = _load(path)
                if payload is None or "autolens_version" not in payload:
                    continue
                folder = "/".join(parts[:-1])
                instrument = (
                    payload.get("instrument")
                    if isinstance(payload.get("instrument"), str)
                    else None
                )
                if section == "runtime":
                    cell = (
                        folder
                        if (instrument and folder.endswith("/" + instrument))
                        else (f"{folder}/{instrument}" if instrument else folder)
                    )
                else:
                    cell = f"{folder}/{m['script']}" + (f"/{instrument}" if instrument else "")
                add(
                    (section, cell, m["config"], bool(m["sparse"])),
                    _point(payload, str(payload["autolens_version"]), str(rel)),
                )
    return raw


# ---------------------------------------------------------------------------
# Qualification, drift, series
# ---------------------------------------------------------------------------


def is_reference_host_class(config: str) -> bool:
    """True when ``config`` is measured on a declared reference host class (``hpc_*``)."""
    return config.startswith(REFERENCE_HOST_CLASS_PREFIXES)


def has_repeat_summary(point: dict) -> bool:
    """True when the compared metric carries a summary of >= 2 independent repeats."""
    n = point.get(REPEAT_SUMMARY_FIELD)
    return isinstance(n, int) and not isinstance(n, bool) and n >= 2


def qualify(point: dict, config: str, conf: dict) -> tuple[bool, str | None, bool]:
    """(qualified, reason, refused) for one point under the release-sweep pin.

    Refused above the load-average cap. Otherwise unqualified, with the first failing reason, when
    the row has no provenance block, is not on a reference host class (laptop rows never qualify
    as trend points), carries no load average or no host, is an HPC row off the pinned node, or
    records a warm-up that never settled (P3, fix phase 6).
    """
    cap = conf.get("loadavg_cap")
    node = conf.get("node")
    if cap is not None and point["loadavg"] is not None and point["loadavg"] > cap:
        return False, f"loadavg {point['loadavg']:.1f} above the cap {cap:g}", True
    if not point["has_provenance"]:
        return False, "no provenance block (pre-#342 row)", False
    if not is_reference_host_class(config):
        return (
            False,
            "not a reference host class; laptop rows never qualify as trend points",
            False,
        )
    if point["loadavg"] is None:
        return False, "provenance carries no load average", False
    if point["host"] is None:
        return False, "provenance carries no host", False
    if node and point["host"] != node:
        return False, f"off the reference host ({point['host']} != {node})", False
    if point.get("warmup_unsettled"):
        return False, point["warmup_unsettled"], False
    return True, None, False


def marker_verdict(marker: dict, config: str, conf: dict) -> dict:
    """How one CPU ``.unusable.json`` marker renders: ``{gpu_only, label, reason}`` (P9).

    A marker is one wall-clock observation. It renders ``GPU-only`` only when it qualifies under
    the same rules as a trend point (:func:`qualify`, so :func:`is_reference_host_class`, a
    recorded host and load average, the load cap and the pinned node). The load judged is the
    larger of the loads recorded at the start of the run and at the timeout. A marker written
    before fix phase 5 records neither host nor load, so it is inconclusive.
    """
    reason_text = str(marker.get("reason") or "")
    timed_out = (
        marker.get("outcome") == "timeout"
        or _finite(marker.get("timeout_seconds")) is not None
        or "timeout" in reason_text
        or "wall-clock" in reason_text
    )
    label = MARKER_TIMED_OUT if timed_out else MARKER_NOT_FINISHED
    host = marker.get("host") if isinstance(marker.get("host"), str) and marker["host"] else None
    loads = [
        v
        for v in (_finite(marker.get(k)) for k in ("loadavg_at_start", "loadavg_at_timeout"))
        if v is not None
    ]
    if host is None and not loads:
        return {
            "gpu_only": False,
            "label": label,
            "reason": "marker records no host or load average (written before fix phase 5)",
        }
    point = {"loadavg": max(loads) if loads else None, "host": host, "has_provenance": True}
    ok, why, _refused = qualify(point, config, conf)
    if not ok:
        return {"gpu_only": False, "label": label, "reason": why}
    return {"gpu_only": True, "label": MARKER_GPU_ONLY, "reason": None}


def drift(points: list[dict]) -> dict:
    """Compare the last two releases' single-jit per call, on the same estimator only."""
    usable = [p for p in points if p["single_jit_s"] is not None]
    if len(usable) < 2:
        return {
            "status": "single-release" if usable else "no-data",
            "ratio": None,
            "from": None,
            "to": None,
        }
    prev, last = usable[-2], usable[-1]
    if headline_estimator(prev) != headline_estimator(last):
        return {
            "status": "estimator-mismatch",
            "ratio": None,
            "from": prev["version"],
            "to": last["version"],
        }
    ratio = last["single_jit_s"] / prev["single_jit_s"] if prev["single_jit_s"] > 0 else None
    delta = abs(last["single_jit_s"] - prev["single_jit_s"])
    if ratio is None:
        status = "no-data"
    elif ratio >= DRIFT_RATIO and delta >= DRIFT_FLOOR_S:
        status = "drifted"
    elif ratio <= 1.0 / DRIFT_RATIO and delta >= DRIFT_FLOOR_S:
        status = "improved"
    else:
        status = "steady"
    return {
        "status": status,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "from": prev["version"],
        "to": last["version"],
    }


def build_series(root: Path, conf: dict) -> tuple[list[dict], list[dict]]:
    """Series (one per key, points de-duplicated per version) and the refused rows."""
    raw = scan(root)
    series: list[dict] = []
    refused: list[dict] = []
    for key in sorted(raw, key=lambda k: (SECTIONS.index(k[0]), k[1], _config_rank(k[2]), k[3])):
        section, cell, config, sparse = key
        by_version: dict[str, dict] = {}
        for point in raw[key]:
            ok, reason, is_refused = qualify(point, config, conf)
            point = dict(point, qualified=ok, reason=reason)
            if is_refused:
                refused.append(
                    dict(point, section=section, cell=cell, config=config, sparse=sparse)
                )
                continue
            cur = by_version.get(point["version"])
            # One point per release: prefer a provenance-carrying row, then the first seen.
            if cur is None or (point["has_provenance"] and not cur["has_provenance"]):
                by_version[point["version"]] = point
        if not by_version:
            continue
        points = [by_version[v] for v in sorted(by_version, key=_version_key)]
        series.append(
            {
                "key": f"{section}:{cell}:{config}{'_sparse' if sparse else ''}",
                "section": section,
                "cell": cell,
                "config": config,
                "sparse": sparse,
                "points": points,
                "drift": drift(points),
            }
        )
    return series, refused


def _config_rank(config: str) -> int:
    return CONFIG_ORDER.index(config) if config in CONFIG_ORDER else len(CONFIG_ORDER)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def build_state(series: list[dict], generated: str) -> dict:
    """The project badge feed (PyAutoBrain/board/state_schema.json, v1)."""
    cells = {s["cell"] for s in series}
    versions = {p["version"] for s in series for p in s["points"]}
    drifted = [s for s in series if s["drift"]["status"] == "drifted"]
    if not series:
        status, headline = "grey", "no result JSON carries a per-call headline yet"
    elif drifted:
        status = "yellow"
        headline = f"{len(drifted)} of {len(series)} series drifted >= {DRIFT_RATIO:g}x at the last release"
    else:
        status = "green"
        headline = (
            f"{len(series)} series across {len(cells)} cells and {len(versions)} releases; "
            f"none drifted >= {DRIFT_RATIO:g}x (single-sample endpoints: not a measured null)"
        )
    items = []
    for s in drifted:
        d = s["drift"]
        text = f"{s['section']}/{s['cell']} {s['config']}{' sparse' if s['sparse'] else ''}: {d['ratio']}x from {d['from']} to {d['to']}"
        items.append(
            {
                "severity": "yellow",
                "text": text,
                "url": f"{REPO_URL}/blob/main/{s['points'][-1]['source'].split('#')[0]}",
                "prompt": f"/profiling triage {s['cell']} {s['config']} — {d['ratio']}x from {d['from']} to {d['to']} (dashboard drift badge)",
            }
        )
    return {
        "schema_version": 1,
        "organ": "autolens_profiling",
        "repo": "autolens_profiling",
        "status": status,
        "headline": headline,
        "updated": generated,
        "pages_url": PAGES_URL,
        "items": items,
    }


# --- profiling-summary v1 (the PyAutoPulse read contract) -------------------


def _producer_revision(root: Path) -> tuple[str | None, str | None]:
    """HEAD of this checkout at render time, or (None, why).

    This is the revision of the producer code that generated the file -- deliberately NOT
    the commit that will first contain the file (which cannot know its own hash). ``--check``
    reuses the committed value, so only the data can make the page stale.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "git unavailable at render"
    sha = proc.stdout.strip()
    if proc.returncode != 0 or not _SHA_RE.match(sha):
        return None, "not a git checkout at render"
    return sha, None


def _release_date(version: str) -> str | None:
    """``2026.9.27.1`` -> ``2026-09-27`` (PyAutoLens release versions encode their date)."""
    parts = version.split(".")
    if len(parts) < 3:
        return None
    try:
        return _dt.date(int(parts[0]), int(parts[1]), int(parts[2])).isoformat()
    except ValueError:
        return None


def _config_identity(config: str) -> dict:
    """``hpc_a100_mp`` -> tier / device / precision; an unknown grammar stays null + reason."""
    parts = config.split("_")
    out = {"tier": None, "device": None, "precision": None, "reason": None}
    if len(parts) >= 3 and parts[-1] in ("fp64", "mp") and parts[0] in ("local", "hpc"):
        out["tier"] = parts[0]
        out["device"] = "_".join(parts[1:-1])
        out["precision"] = "float64" if parts[-1] == "fp64" else "mixed"
    else:
        out["reason"] = f"config label {config!r} is outside the sweep grammar"
    return out


def _evidence(source: str) -> dict:
    """A point's ``source`` -> repo-relative path + the in-file fragment (a comparison config)."""
    path, _, fragment = source.partition("#")
    return {"path": path, "fragment": fragment or None}


def _summary_record(s: dict, p: dict) -> dict:
    ident = _config_identity(s["config"])
    return {
        "id": f"{s['key']}@{p['version']}",
        "axis": "runtime",
        "unit": "s",
        "measurement": {
            "single_jit_s": p["single_jit_s"],
            "vmap_per_call_s": p["vmap_per_call_s"],
            # Only where the headline is the steady median (P8): the legacy block mean beside it.
            **(
                {"single_jit_block_mean_s": p.get("single_jit_block_mean_s")}
                if headline_estimator(p) == ESTIMATOR_MEDIAN
                else {}
            ),
        },
        "identity": {
            "section": s["section"],
            "cell": s["cell"],
            "config": s["config"],
            "sparse": s["sparse"],
            "tier": ident["tier"],
            "device": ident["device"],
            "backend": p["backend"],
            "precision": ident["precision"],
            "library": "PyAutoLens",
            "library_version": p["version"],
            "release_date": _release_date(p["version"]),
            "reason": ident["reason"],
        },
        "provenance": {
            "host": p["host"],
            "job": p["job"],
            "loadavg": p["loadavg"],
            "has_provenance": p["has_provenance"],
            "qualified": p["qualified"],
            "reason": p["reason"],
        },
        "evidence": _evidence(p["source"]),
    }


_COMPARISON_STATUS = {
    "drifted": "drifted",
    "improved": "improved",
    "steady": "flat",
    "single-release": "insufficient",
    "no-data": "insufficient",
    "estimator-mismatch": "insufficient",
}


def _summary_comparison(s: dict) -> dict:
    """The producer's own drift verdict for one series -- displayed by the organ, never redone.

    Inside the 2x band (``steady``) is published as ``flat`` only when both endpoints carry a
    repeat summary; with a single-sample endpoint it is ``insufficient``. ``drifted`` /
    ``improved`` keep their status (gross-band signals) with a single-sample caveat (human
    decision 2026-10-08, autolens_profiling#362).
    """
    d = s["drift"]
    status = _COMPARISON_STATUS[d["status"]]
    reasons: list[str] = []
    if d["status"] == "single-release":
        reasons.append("one release only")
    elif d["status"] == "no-data":
        reasons.append("no single-jit headline in the last two releases")
    elif d["status"] == "estimator-mismatch":
        reasons.append(ESTIMATOR_MISMATCH_REASON)
    by_version = {p["version"]: p for p in s["points"]}
    endpoints = [by_version.get(v) for v in (d["from"], d["to"]) if v is not None]
    if d["status"] in ("steady", "drifted", "improved"):
        single = len(endpoints) < 2 or not all(
            p is not None and has_repeat_summary(p) for p in endpoints
        )
        if d["status"] == "steady":
            if single:
                status = "insufficient"
                reasons.append(SINGLE_SAMPLE_NULL_REASON)
            else:
                reasons.append(FLAT_BAND_REASON)
        elif single:
            reasons.append(SINGLE_SAMPLE_REASON)
    for role, p in zip(("baseline", "candidate"), endpoints):
        if p is not None and not p["qualified"]:
            reasons.append(f"{role} row unqualified: {p['reason']}")
    return {
        "comparison_key": s["key"],
        "policy": DRIFT_POLICY,
        "axis": "runtime",
        "metric": "single_jit_s",
        "baseline": d["from"],
        "candidate": d["to"],
        "ratio": d["ratio"],
        "status": status,
        "qualified": bool(endpoints)
        and len(endpoints) == 2
        and all(p["qualified"] for p in endpoints),
        "reasons": reasons,
    }


def build_summary(
    series: list[dict],
    refused: list[dict],
    conf: dict,
    generated: str,
    revision: str | None,
) -> dict:
    """The ``profiling-summary`` v1 feed (``dashboard/README.md``)."""
    records = [_summary_record(s, p) for s in series for p in s["points"]]
    comparisons = [_summary_comparison(s) for s in series]
    excluded = [
        {
            "id": f"{r['section']}:{r['cell']}:{r['config']}{'_sparse' if r['sparse'] else ''}@{r['version']}",
            "reason": r["reason"],
            "evidence": _evidence(r["source"]),
        }
        for r in refused
    ]
    release_dates = sorted(
        {d for d in (_release_date(r["identity"]["library_version"]) for r in records) if d}
    )
    if release_dates:
        evidence_updated_at = f"{release_dates[-1]}T00:00:00Z"
        evidence_reason = None
    else:
        evidence_updated_at = None
        evidence_reason = "no included measurement" if not records else "no parseable release date"
    limitations = [
        "runtime axis only: compile times, VRAM and per-step component timings are not in this feed",
        "evidence_updated_at is the newest release date encoded in a measured library version; "
        "result rows do not record measurement wall-clock time",
        "valid_until is null: this project declares no freshness policy for release trends",
        "vmap_per_call_s is null where a row carries no vmap block",
    ]
    limitations.append(
        "only reference-host-class rows (hpc_*) can be qualified: laptop rows never qualify as "
        "trend points, and a provenance block without a load average or host is unqualified"
    )
    if any(
        reason in (SINGLE_SAMPLE_NULL_REASON, SINGLE_SAMPLE_REASON)
        for c in comparisons
        for reason in c["reasons"]
    ):
        limitations.append(
            "comparison endpoints without a repeat summary are single samples (one 10-call block "
            "mean): a ratio inside the 2x band is insufficient, not flat; drifted / improved are "
            "gross-band flags with a single-sample caveat"
        )
    if any("single_jit_block_mean_s" in r["measurement"] for r in records):
        limitations.append(
            "single_jit_s is the steady median (>= 5 warm calls, median of individually timed "
            "calls) where a record also carries single_jit_block_mean_s, else the legacy "
            "headline; a comparison whose endpoints use different estimators is insufficient"
        )
    if any(not r["provenance"]["has_provenance"] for r in records):
        limitations.append(
            "rows without a device.provenance block (pre-autolens_profiling#342) are unqualified"
        )
    if conf.get("node") and any(
        r["provenance"]["reason"] and "off the reference host" in r["provenance"]["reason"]
        for r in records
    ):
        limitations.append(
            f"HPC rows measured off the reference host {conf['node']} are unqualified"
        )
    if excluded:
        limitations.append(
            f"{len(excluded)} row(s) above the load-average cap {conf.get('loadavg_cap'):g} are "
            "listed under coverage.excluded and never enter records"
        )
    if revision is None:
        limitations.append(
            "producer_revision is null: not a git checkout, or git unavailable, at render"
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "version": SUMMARY_VERSION,
        "project": SUMMARY_PROJECT,
        "scope": SUMMARY_SCOPE,
        "generated_at": generated,
        "evidence_updated_at": evidence_updated_at,
        "evidence_updated_at_reason": evidence_reason,
        "valid_until": None,
        "producer_revision": revision,
        "comparison_policy": {
            "id": DRIFT_POLICY,
            "ratio": DRIFT_RATIO,
            "floor_s": DRIFT_FLOOR_S,
            "reference_host": conf.get("node"),
            "loadavg_cap": conf.get("loadavg_cap"),
        },
        "coverage": {
            "expected": {
                "sections": list(SECTIONS),
                "cells": None,
                "reason": "no declared cell matrix; the results/ scan is the inventory",
            },
            "observed": {
                "series": len(series),
                "points": len(records),
                "releases": len({r["identity"]["library_version"] for r in records}),
                "cells": len({s["cell"] for s in series}),
            },
            "excluded": excluded,
        },
        "records": records,
        "comparisons": comparisons,
        "limitations": limitations,
    }


def _safe_relative_path(path) -> bool:
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return False
    return ".." not in path.split("/")


def _finite_or_null(v) -> bool:
    return v is None or (
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    )


def validate_summary(payload: dict) -> list[str]:
    """Exchange-contract findings for a ``profiling-summary`` payload (empty = valid).

    This is the producer's own guard: it refuses to publish what the organ would reject --
    unknown schema/version, missing fields, duplicate ids, non-finite numbers, bad dates,
    unsafe evidence paths. Domain meaning (what a series is, when it drifted) is not judged
    here; that stays in build_series/drift.
    """
    out: list[str] = []
    if not isinstance(payload, dict):
        return ["payload is not an object"]
    if payload.get("schema") != SUMMARY_SCHEMA:
        out.append(f"schema {payload.get('schema')!r} != {SUMMARY_SCHEMA!r}")
    if payload.get("version") != SUMMARY_VERSION:
        out.append(f"version {payload.get('version')!r} != {SUMMARY_VERSION}")
    for key in (
        "project",
        "scope",
        "generated_at",
        "evidence_updated_at",
        "valid_until",
        "producer_revision",
        "coverage",
        "records",
        "comparisons",
        "limitations",
    ):
        if key not in payload:
            out.append(f"missing required field {key!r}")
    for key in ("generated_at", "evidence_updated_at", "valid_until"):
        v = payload.get(key)
        if v is not None and not (isinstance(v, str) and _ISO_UTC_RE.match(v)):
            out.append(f"{key} is not an ISO-8601 UTC timestamp: {v!r}")
    if not isinstance(payload.get("generated_at"), str):
        out.append("generated_at must be a timestamp, not null")
    vu, ga = payload.get("valid_until"), payload.get("generated_at")
    if isinstance(vu, str) and isinstance(ga, str) and vu < ga:
        out.append("valid_until precedes generated_at")
    rev = payload.get("producer_revision")
    if rev is not None and not (isinstance(rev, str) and _SHA_RE.match(rev)):
        out.append(f"producer_revision is not a 40-hex commit or null: {rev!r}")
    records = payload.get("records")
    if not isinstance(records, list):
        out.append("records is not a list")
        records = []
    seen: set[str] = set()
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            out.append(f"records[{i}] is not an object")
            continue
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            out.append(f"records[{i}] has no id")
        elif rid in seen:
            out.append(f"duplicate record id {rid!r}")
        else:
            seen.add(rid)
        meas = r.get("measurement") if isinstance(r.get("measurement"), dict) else {}
        values = [meas.get("single_jit_s"), meas.get("vmap_per_call_s")]
        if not all(_finite_or_null(v) for v in values):
            out.append(f"records[{i}] carries a non-finite measurement")
        if all(v is None for v in values):
            out.append(f"records[{i}] carries no measurement")
        ev = r.get("evidence") if isinstance(r.get("evidence"), dict) else {}
        if not _safe_relative_path(ev.get("path")):
            out.append(
                f"records[{i}] evidence path is not a safe repo-relative path: {ev.get('path')!r}"
            )
        prov = r.get("provenance") if isinstance(r.get("provenance"), dict) else {}
        if not _finite_or_null(prov.get("loadavg")):
            out.append(f"records[{i}] carries a non-finite loadavg")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list):
        out.append("comparisons is not a list")
        comparisons = []
    seen_cmp: set[str] = set()
    for i, c in enumerate(comparisons):
        if not isinstance(c, dict):
            out.append(f"comparisons[{i}] is not an object")
            continue
        key = c.get("comparison_key")
        if not isinstance(key, str) or not key:
            out.append(f"comparisons[{i}] has no comparison_key")
        elif key in seen_cmp:
            out.append(f"duplicate comparison_key {key!r}")
        else:
            seen_cmp.add(key)
        if c.get("status") not in set(_COMPARISON_STATUS.values()):
            out.append(f"comparisons[{i}] status {c.get('status')!r} is not in the contract")
        if not _finite_or_null(c.get("ratio")):
            out.append(f"comparisons[{i}] carries a non-finite ratio")
        if c.get("policy") != DRIFT_POLICY:
            out.append(f"comparisons[{i}] names an unknown policy {c.get('policy')!r}")
    cov = payload.get("coverage")
    if isinstance(cov, dict):
        obs = cov.get("observed") if isinstance(cov.get("observed"), dict) else None
        if obs is None:
            out.append("coverage.observed missing")
        else:
            for k, v in obs.items():
                if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
                    out.append(f"coverage.observed.{k} is not a non-negative integer")
        for i, x in enumerate(cov.get("excluded") or []):
            ev = (
                x.get("evidence")
                if isinstance(x, dict) and isinstance(x.get("evidence"), dict)
                else {}
            )
            if not _safe_relative_path(ev.get("path")):
                out.append(f"coverage.excluded[{i}] evidence path is not safe: {ev.get('path')!r}")
    elif "coverage" in payload:
        out.append("coverage is not an object")
    if "limitations" in payload and not (
        isinstance(payload["limitations"], list)
        and all(isinstance(x, str) and x for x in payload["limitations"])
    ):
        out.append("limitations must be a list of non-empty strings")
    return out


# --- HTML -------------------------------------------------------------------

# Categorical slots from the dataviz reference palette, fixed order, one per sweep config.
_PALETTE_LIGHT = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
_PALETTE_DARK = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)

_CSS = """
:root { color-scheme: light dark;
  --surface: #fcfcfb; --surface-2: #f0efec; --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8985; --line: #d9d8d3;
  --good: #008300; --warn: #b26a00; --bad: #c42b2b; --accent: #2a78d6;
  %(light)s }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface: #1a1a19; --surface-2: #262625; --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e88; --line: #3a3a37;
  --good: #34c759; --warn: #e0a030; --bad: #ff6b6b; --accent: #3987e5; %(dark)s } }
:root[data-theme="dark"] {
  --surface: #1a1a19; --surface-2: #262625; --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e88; --line: #3a3a37;
  --good: #34c759; --warn: #e0a030; --bad: #ff6b6b; --accent: #3987e5; %(dark)s }
* { box-sizing: border-box; }
body { margin: 0; padding: 0 16px 48px; background: var(--surface); color: var(--ink);
  font: 15px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; overflow-wrap: anywhere; }
header { padding: 20px 0 8px; border-bottom: 1px solid var(--line); }
h1 { font-size: 22px; margin: 0 0 4px; }
h1 span { color: var(--accent); }
h2 { font-size: 17px; margin: 28px 0 8px; color: var(--accent); }
.meta { color: var(--ink-2); font-size: 13px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; font-weight: 600;
  border: 1px solid var(--line); color: var(--ink-2); vertical-align: middle; }
.badge.drifted { color: var(--bad); border-color: var(--bad); }
.badge.improved, .badge.green { color: var(--good); border-color: var(--good); }
.badge.yellow { color: var(--warn); border-color: var(--warn); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
.panel { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: var(--surface); }
.panel h3 { font-size: 14px; margin: 0 0 6px; font-weight: 600; }
.panel h3 code { color: var(--accent); font-size: 13px; }
svg { width: 100%%; height: auto; display: block; }
.axis text { fill: var(--ink-3); font-size: 10px; }
.axis line, .axis path { stroke: var(--line); }
.grid-line { stroke: var(--line); stroke-dasharray: 2 3; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 6px 0 0; padding: 0; list-style: none; font-size: 12px; color: var(--ink-2); }
.legend i { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 5px; border-radius: 2px; }
details { margin-top: 6px; font-size: 12px; }
details summary { cursor: pointer; color: var(--ink-2); }
table { border-collapse: collapse; width: 100%%; margin-top: 4px; }
th, td { text-align: left; padding: 2px 6px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; }
.muted { color: var(--ink-3); }
.hollow { fill: var(--surface); }
.refused td { color: var(--ink-3); }
footer { margin-top: 32px; color: var(--ink-3); font-size: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
"""


def _fmt_s(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1.0:
        return f"{v:.2f} s"
    return f"{v * 1000:.2f} ms"


def _fmt_axis(v: float) -> str:
    """Compact power-of-ten tick: 10 ms, 100 ms, 1 s, 10 s."""
    return f"{v:g} s" if v >= 1.0 else f"{v * 1000:g} ms"


def _svg_panel(cell_series: list[dict], versions: list[str], slots: dict[str, int]) -> str:
    """One small-multiple: x = release (categorical, shared order), y = per call, log scale."""
    w, h, ml, mr, mt, mb = 320, 170, 50, 8, 8, 30
    pw, ph = w - ml - mr, h - mt - mb
    values = [p["single_jit_s"] for s in cell_series for p in s["points"] if p["single_jit_s"]]
    if not values:
        return '<p class="muted">no per-call headline in these rows</p>'
    lo, hi = min(values), max(values)
    lo_e, hi_e = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    if hi_e == lo_e:
        hi_e += 1

    def x_of(version: str) -> float:
        i = versions.index(version)
        return ml + (pw * (i + 0.5) / len(versions))

    def y_of(v: float) -> float:
        return mt + ph * (1 - (math.log10(v) - lo_e) / (hi_e - lo_e))

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="per-call run time per release">']
    out.append('<g class="axis">')
    for e in range(lo_e, hi_e + 1):
        y = y_of(10**e)
        out.append(f'<line class="grid-line" x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(
            f'<text x="{ml - 4}" y="{y + 3:.1f}" text-anchor="end">{_fmt_axis(10**e)}</text>'
        )
    for v in versions:
        x = x_of(v)
        out.append(
            f'<text x="{x:.1f}" y="{h - 14}" text-anchor="middle" transform="rotate(0)">{html.escape(v.replace("2026.", ""))}</text>'
        )
    out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{mt + ph}" y2="{mt + ph}"/>')
    out.append("</g>")
    for s in cell_series:
        color = f"var(--series-{slots[s['config']]})"
        pts = [
            (x_of(p["version"]), y_of(p["single_jit_s"]), p)
            for p in s["points"]
            if p["single_jit_s"]
        ]
        if len(pts) > 1:
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts)
            )
            dash = ' stroke-dasharray="4 3"' if s["sparse"] else ""
            out.append(
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"{dash}/>'
            )
        for x, y, p in pts:
            tip = html.escape(
                f"{s['config']}{' sparse' if s['sparse'] else ''} @ {p['version']}: {_fmt_s(p['single_jit_s'])} per call"
                + (f" ({p['headline_note']})" if p.get("headline_note") else "")
                + (
                    f" ({ESTIMATOR_MEDIAN}; block mean {_fmt_s(p['single_jit_block_mean_s'])})"
                    if headline_estimator(p) == ESTIMATOR_MEDIAN
                    and p.get("single_jit_block_mean_s") is not None
                    else ""
                )
                + (
                    f", steady median {_fmt_s(p['single_jit_median_s'])}"
                    if p.get("single_jit_median_s") is not None
                    and headline_estimator(p) != ESTIMATOR_MEDIAN
                    else ""
                )
                + (f", vmap {_fmt_s(p['vmap_per_call_s'])}" if p["vmap_per_call_s"] else "")
                + (f" — {p['host']}" if p["host"] else "")
                + (f", job {p['job']}" if p["job"] else "")
                + (f", load {p['loadavg']:.1f}" if p["loadavg"] is not None else "")
                + (f" — {p['reason']}" if p.get("reason") else "")
            )
            cls = "" if p["qualified"] else ' class="hollow"'
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="{color}" stroke-width="2"{cls}>'
                f"<title>{tip}</title></circle>"
            )
    out.append("</svg>")
    return "\n".join(out)


def render_html(
    series: list[dict], refused: list[dict], conf: dict, generated: str, state: dict
) -> str:
    versions = sorted({p["version"] for s in series for p in s["points"]}, key=_version_key)
    slots = {c: i + 1 for i, c in enumerate(CONFIG_ORDER)}
    css = _CSS % {
        "light": " ".join(f"--series-{i + 1}: {c};" for i, c in enumerate(_PALETTE_LIGHT)),
        "dark": " ".join(f"--series-{i + 1}: {c};" for i, c in enumerate(_PALETTE_DARK)),
    }
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>PyAutoLens run time over time</title>",
        f"<style>{css}</style></head><body>",
        "<header>",
        "<h1>PyAuto <span>Profiling</span> — run time over time</h1>",
        f'<p class="meta">Per-call likelihood cost per PyAutoLens release, one panel per cell, one line per sweep config. '
        f'<span class="badge {state["status"]}">{html.escape(state["status"])}</span> {html.escape(state["headline"])}. '
        f"Rendered {html.escape(generated)} from <code>results/</code> by <code>build_dashboard.py</code>; "
        f"reference host <code>{html.escape(conf.get('node') or 'unset')}</code>, load-average cap {conf.get('loadavg_cap') or '—'}. "
        "Hollow markers are rows without a provenance block or off the reference host; refused rows are listed at the foot. "
        f"A GPU per-call value headlined by <code>full_pipeline_single_jit</code> is labelled <em>{FIRST_BLOCK_NOTE}</em>: "
        "it is the mean of the 10-call block right after the first call, which on the A100 can sit in the post-compile transient. "
        "Rows that carry it also show the steady median (&ge; 5 warm calls, median of individually timed calls). "
        f"Where a row records that median it is the headline, labelled <em>{ESTIMATOR_MEDIAN}</em>, with the block mean beside it; "
        "drift never compares a median with a block mean. "
        "Committed rows are not re-based. "
        "This page renders and holds timing history; it judges no lever "
        '(<a href="' + REPO_URL + '/blob/main/wiki/index.md">campaign index</a>).</p>',
        "</header>",
    ]
    if not series:
        parts.append('<p class="muted">Nothing to plot yet.</p>')
    for section in SECTIONS:
        cells: dict[str, list[dict]] = {}
        for s in series:
            if s["section"] == section:
                cells.setdefault(s["cell"], []).append(s)
        if not cells:
            continue
        parts.append(f'<h2>{section}</h2><div class="grid">')
        for cell, cell_series in cells.items():
            worst = (
                "drifted"
                if any(s["drift"]["status"] == "drifted" for s in cell_series)
                else (
                    "improved"
                    if any(s["drift"]["status"] == "improved" for s in cell_series)
                    else None
                )
            )
            badge = f' <span class="badge {worst}">{worst}</span>' if worst else ""
            parts.append(f'<div class="panel"><h3><code>{html.escape(cell)}</code>{badge}</h3>')
            parts.append(_svg_panel(cell_series, versions, slots))
            parts.append('<ul class="legend">')
            for s in cell_series:
                label = s["config"] + (" (sparse, dashed)" if s["sparse"] else "")
                parts.append(
                    f'<li><i style="background: var(--series-{slots[s["config"]]})"></i>{html.escape(label)}</li>'
                )
            parts.append("</ul>")
            parts.append(
                "<details><summary>table</summary><table><tr><th>config</th><th>release</th><th>per call</th><th>vmap</th><th>host / job / load</th><th>drift</th></tr>"
            )
            for s in cell_series:
                for p in s["points"]:
                    prov = (
                        " / ".join(
                            str(x)
                            for x in (
                                p["host"],
                                p["job"],
                                f"{p['loadavg']:.1f}" if p["loadavg"] is not None else None,
                            )
                            if x
                        )
                        or "—"
                    )
                    note = (
                        f' <span class="muted">({html.escape(p["reason"])})</span>'
                        if p.get("reason")
                        else ""
                    )
                    d = s["drift"]
                    dtxt = (
                        f"{'within 2x band' if d['status'] == 'steady' else d['status']} {d['ratio']}x"
                        if d["ratio"] and p is s["points"][-1]
                        else ""
                    )
                    parts.append(
                        f"<tr><td>{html.escape(s['config'])}{' sparse' if s['sparse'] else ''}</td><td>{html.escape(p['version'])}</td>"
                        f"<td>{_per_call_html(p)}</td><td>{_fmt_s(p['vmap_per_call_s'])}</td><td>{html.escape(prov)}{note}</td><td>{html.escape(dtxt)}</td></tr>"
                    )
            parts.append("</table></details></div>")
        parts.append("</div>")
    if refused:
        parts.append(
            '<h2>refused rows</h2><p class="meta">Above the load-average cap; kept in the tree, not a trend point.</p>'
        )
        parts.append(
            "<table><tr><th>cell</th><th>config</th><th>release</th><th>per call</th><th>host / job / load</th><th>source</th></tr>"
        )
        for r in refused:
            parts.append(
                f'<tr class="refused"><td>{html.escape(r["section"] + "/" + r["cell"])}</td><td>{html.escape(r["config"])}{" sparse" if r["sparse"] else ""}</td>'
                f"<td>{html.escape(r['version'])}</td><td>{_fmt_s(r['single_jit_s'])}</td>"
                f"<td>{html.escape(str(r['host']))} / {html.escape(str(r['job']))} / {r['loadavg']:.1f}</td><td><code>{html.escape(r['source'])}</code></td></tr>"
            )
        parts.append("</table>")
    parts.append(
        "<footer>Data: the versioned summaries, config-tagged rows and sweep comparisons under "
        f'<a href="{REPO_URL}/tree/main/results">results/</a>; the run-time drift badge uses the profiling conductor\'s '
        f"{DRIFT_RATIO:g}x ratio with a {DRIFT_FLOOR_S * 1000:g} ms floor. Refresh: <code>profile.yml</code> on release tags. "
        f'Feed: <a href="state.json">state.json</a> · data: <a href="series.json">series.json</a>.</footer>'
    )
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def build(
    root: Path, generated: str | None = None, revision: str | None | object = ...
) -> dict[str, str]:
    """Render every output as text, keyed by filename under dashboard/.

    ``revision`` is the producer revision the summary records: ``...`` (default) reads HEAD,
    ``None`` records null, a string is taken as given (``--check`` passes the committed one).
    """
    root = root.resolve()
    conf = read_release_sweep_conf(root)
    generated = generated or _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    series, refused = build_series(root, conf)
    state = build_state(series, generated)
    if revision is ...:
        revision, _why = _producer_revision(root)
    summary = build_summary(series, refused, conf, generated, revision)
    findings = validate_summary(summary)
    if findings:
        raise ValueError("profiling-summary would be invalid: " + "; ".join(findings))
    series_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated": generated,
        "reference_host": conf.get("node"),
        "loadavg_cap": conf.get("loadavg_cap"),
        "drift_ratio": DRIFT_RATIO,
        "drift_floor_s": DRIFT_FLOOR_S,
        "series": series,
        "refused": refused,
    }
    outputs = {
        "series.json": json.dumps(series_doc, indent=1) + "\n",
        "state.json": json.dumps(state, indent=2) + "\n",
        "summary.json": json.dumps(summary, indent=1) + "\n",
        "index.html": render_html(series, refused, conf, generated, state),
    }
    if (root / "catalogue/registry.json").is_file():
        # The companion v2 catalogue has its own declared scientific matrix.
        # Keep v1 bytes and consumers intact until the browser migration.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_catalogue import render_outputs

        outputs.update(render_outputs(root, generated, revision))
        from setup_page import render as render_setup_page

        outputs["index.html"] = render_setup_page(json.loads(outputs["catalogue.json"]))
        if (root / "catalogue/wiki_bindings.json").is_file():
            from build_setup_wiki import render_outputs as render_wiki

            catalogue = json.loads(outputs["catalogue.json"])
            shards = {
                ref["setup_id"]: json.loads(outputs[ref["path"]])
                for ref in catalogue["evidence_shards"]
            }
            outputs.update(
                {
                    "../" + name: content
                    for name, content in render_wiki(root, catalogue, shards).items()
                }
            )
    return outputs


def _existing_stamp(out_dir: Path) -> str | None:
    doc = _load(out_dir / "series.json")
    stamp = doc.get("generated") if doc else None
    return stamp if isinstance(stamp, str) else None


def _existing_revision(out_dir: Path) -> str | None:
    doc = _load(out_dir / "summary.json")
    rev = doc.get("producer_revision") if doc else None
    return rev if isinstance(rev, str) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if dashboard/ would change")
    parser.add_argument(
        "--root", type=Path, default=None, help="repo root (default: the ruff.toml dir)"
    )
    args = parser.parse_args(argv)
    root = (args.root or _profiling_root()).resolve()
    out_dir = root / "dashboard"

    # --check re-renders with the committed stamp and producer revision so only the data can
    # differ (HEAD moves with every commit; the summary records the revision that rendered it).
    if args.check:
        outputs = build(
            root, generated=_existing_stamp(out_dir), revision=_existing_revision(out_dir)
        )
    else:
        outputs = build(root)
    stale = [
        name
        for name, text in outputs.items()
        if not (out_dir / name).is_file() or (out_dir / name).read_text(encoding="utf-8") != text
    ]
    obsolete = []
    if "catalogue.json" in outputs:
        from build_catalogue import obsolete_shards

        obsolete = obsolete_shards(out_dir, outputs)
        stale.extend(p.relative_to(out_dir).as_posix() for p in obsolete)
        if (root / "catalogue/wiki_bindings.json").is_file():
            from build_setup_wiki import obsolete_pages

            old_pages = obsolete_pages(
                root,
                {
                    name[3:]: content
                    for name, content in outputs.items()
                    if name.startswith("../wiki/")
                },
            )
            stale.extend(str(p.relative_to(root)) for p in old_pages)
            obsolete.extend(old_pages)
    n_series = json.loads(outputs["series.json"])["series"]
    if args.check:
        if stale:
            print(
                f"build_dashboard: STALE — {', '.join('dashboard/' + n for n in stale)} would change; run build_dashboard.py and commit"
            )
            return 1
        print(f"build_dashboard: dashboard/ is current ({len(n_series)} series)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text(text, encoding="utf-8")
    for path in obsolete:
        path.unlink()
    print(
        "build_dashboard: wrote dashboard artifacts (including catalogue when registered) "
        f"({len(n_series)} series)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
