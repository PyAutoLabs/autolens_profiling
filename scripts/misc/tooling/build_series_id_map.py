"""Generate ``catalogue/series_id_map.json``: pre-move dashboard series ids -> v3 ids (stdlib only).

Run from the repo root::

    python scripts/misc/tooling/build_series_id_map.py           # write the map
    python scripts/misc/tooling/build_series_id_map.py --check   # exit 1 if the committed map is stale

Before the day-zero move (autolens_profiling#442) ``build_dashboard.py`` keyed its trend series
on the legacy file grammar (``<section>:<cell>:<config>[_sparse]``, read from
``results/{runtime,breakdown,simulators,lens}``). It now keys them on the v3 identity and reads
the mirrored tree natively. This generator keeps the published history continuous, derived
mechanically from ``catalogue/migration_day_zero.json`` (never hand-written):

1. It replays the pre-move scan over the map's **old** paths (every moved entry plus the
   unresolved rows left in place), with the legacy grammar, to recover every pre-move series
   and the files and versions behind it.
2. Each pre-move series takes the v3 id its mirrored-tree members now publish under
   (``build_dashboard.native_points``); a series with no mirrored member keeps its old id
   behind ``legacy:``. Two old series never share a new id and a series never spans two.
3. ``sources`` attaches every pre-move point the native rules do not reach (archived and
   ``results/misc`` files, the unresolved rows, an ambiguous ``comparison.json`` entry) to its
   series id with the version the legacy grammar read.

``series`` records, per old id, the new id and the releases it held, which
``test_build_dashboard.py`` checks against the built series. ``cells`` maps each pre-move
runtime cell to its current cell id, so ``build_readme.py`` can match archived named baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dashboard as bd  # noqa: E402
import build_migration_map as mm  # noqa: E402

SCHEMA = "autolens-profiling-series-id-map"
VERSION = 1
LEGACY_SECTIONS = ("runtime", "breakdown", "simulators", "lens")


def legacy_files(root: Path) -> list[tuple[str, str]]:
    """``(old path, current path)`` of every result JSON the day-zero map knows, by old path."""
    doc = json.loads((root / mm.MAP_PATH).read_text())
    pairs = {e["old"]: e["new"] for e in doc.get("entries", [])}
    pairs.update({u["old"]: u["old"] for u in doc.get("unresolved", [])})
    return sorted(
        (old, new) for old, new in pairs.items() if old.endswith(".json") and (root / new).is_file()
    )


def legacy_points(root: Path) -> list[tuple[str, str, str, dict]]:
    """``(old series id, version, current source, payload)``: the pre-move scan, replayed.

    The grammar is the pre-move ``build_dashboard.scan`` verbatim, applied to old paths.
    """
    out = []
    files = legacy_files(root)
    for section in LEGACY_SECTIONS:
        prefix = f"results/{section}/"
        for old, new in files:
            if not old.startswith(prefix):
                continue
            legacy = PurePosixPath(old)
            parts = legacy.relative_to(prefix).parts
            payload = bd._load(root / new)
            if payload is None:
                continue
            if legacy.name == "comparison.json":
                configs = payload.get("configs") if section == "runtime" else None
                if not isinstance(configs, dict):
                    continue
                cell = "/".join(parts[:-1])
                for config, entry in configs.items():
                    if not isinstance(entry, dict) or "autolens_version" not in entry:
                        continue
                    name, sparse = (
                        (config[:-7], True) if config.endswith("_sparse") else (config, False)
                    )
                    key = f"{section}:{cell}:{name}{'_sparse' if sparse else ''}"
                    out.append((key, str(entry["autolens_version"]), f"{new}#{config}", entry))
                continue
            m = bd.ARTIFACT_RE.match(legacy.name)
            tagged = bd.CONFIG_TAGGED_RE.match(legacy.name)
            if m:
                instrument, version, config = m["extra"], m["version"], "local_cpu_fp64"
            elif tagged and section in ("runtime", "breakdown"):
                if "autolens_version" not in payload:
                    continue
                m, instrument = tagged, None
                version, config = str(payload["autolens_version"]), tagged["config"]
            else:
                continue
            if instrument is None and isinstance(payload.get("instrument"), str):
                instrument = payload["instrument"]
            folder = "/".join(parts[:-1])
            if section == "runtime":
                cell = (
                    folder
                    if (instrument and folder.endswith("/" + instrument))
                    else (f"{folder}/{instrument}" if instrument else folder)
                )
            else:
                cell = f"{folder}/{m['script']}" + (f"/{instrument}" if instrument else "")
            key = f"{section}:{cell}:{config}{'_sparse' if m['sparse'] else ''}"
            out.append((key, version, new, payload))
    return out


def published(root: Path, points) -> dict[str, list[str]]:
    """Old id -> the releases its published series held (refused rows never published; an id
    whose every row was refused holds none but still needs a new id for its refused rows)."""
    conf = bd.read_release_sweep_conf(root)
    out: dict[str, set] = {}
    for key, version, source, payload in points:
        point = bd._point(payload, version, source)
        if point is None:
            continue
        held = out.setdefault(key, set())
        config = bd.parse_key(bd.LEGACY_PREFIX + key)["config"]
        if not bd.qualify(point, config, conf)[2]:
            held.add(version)
    return {k: sorted(v, key=bd._version_key) for k, v in out.items()}


def comparison_siblings(root: Path) -> dict[str, str]:
    """``<current comparison.json>#<config>`` -> the v3 id of the raw row it summarised.

    Before the move a sweep wrote ``<script>_<config>[_sparse].json`` beside its cell's
    ``comparison.json``; the map says where that raw row now lives, which names the setup
    an entry belongs to even where the label alone is ambiguous (several setups share it).
    """
    files = legacy_files(root)
    current = dict(files)
    out: dict[str, set] = {}
    for old, new in files:
        tagged = bd.CONFIG_TAGGED_RE.match(old.rsplit("/", 1)[-1])
        facts = bd.lc.mirrored(new)
        if not tagged or facts is None or facts["location"] != "cell":
            continue
        comparison = old.rsplit("/", 1)[0] + "/comparison.json"
        if comparison not in current or facts["file_measurement"] not in bd.SERIES_MEASUREMENTS:
            continue
        config = tagged["config"] + ("_sparse" if tagged["sparse"] else "")
        out.setdefault(f"{current[comparison]}#{config}", set()).add(bd.series_key(facts))
    return {k: v.pop() for k, v in out.items() if len(v) == 1}


def runtime_cells(root: Path) -> dict[str, str]:
    """Pre-move runtime cell (``imaging/pixelization/hst``) -> its current cell id.

    From each moved ``results/runtime/<cell>/comparison.json``: a mirrored cell is
    ``<dataset_type>/<variant>/<instrument>``, a misc one its path under
    ``results/misc/likelihood_runtime/``. ``build_readme.py`` matches archived named
    baselines (recorded under the pre-move cell) to the current row with it.
    """
    out = {}
    misc = "results/misc/likelihood_runtime/"
    for old, new in legacy_files(root):
        if not (old.startswith("results/runtime/") and old.endswith("/comparison.json")):
            continue
        cell = old[len("results/runtime/") : -len("/comparison.json")]
        if new.startswith(misc):
            out[cell] = new[len(misc) : -len("/comparison.json")]
        elif bd._comparison_cell(new):
            out[cell] = "/".join(bd._comparison_cell(new))
    return dict(sorted(out.items()))


def build(root: Path) -> dict:
    root = root.resolve()
    points = legacy_points(root)
    siblings = comparison_siblings(root)
    index = bd._cell_index(bd.result_files(root))
    native: dict[str, dict[str, tuple[str, str]]] = {}
    for rel in {s.partition("#")[0] for _, _, s, _ in points}:
        native[rel] = {
            src: (key, version) for key, version, src, _ in bd.native_points(root, rel, index)
        }
    members: dict[str, set] = {}
    for key, version, source, _ in points:
        hit = native[source.partition("#")[0]].get(source)
        if hit is not None and hit[1] == version:
            members.setdefault(key, set()).add(hit[0])
        elif hit is None and source in siblings:
            members.setdefault(key, set()).add(siblings[source])
    releases = published(root, points)
    series, owner, problems = {}, {}, []
    for old in sorted(releases):
        ids = members.get(old, set())
        if len(ids) > 1:
            problems.append(f"{old} spans {sorted(ids)}")
        new = min(ids) if ids else bd.LEGACY_PREFIX + old
        if new in owner:
            problems.append(f"{old} and {owner[new]} both map to {new}")
        owner[new] = old
        series[old] = {"id": new, "releases": releases[old]}
    sources = {}
    for key, version, source, _ in points:
        if key not in series:
            continue
        target = series[key]["id"]
        if native[source.partition("#")[0]].get(source) != (target, version):
            sources[source] = {"id": target, "version": version}
    if problems:
        raise SystemExit("series id map is not one-to-one:\n  " + "\n  ".join(problems))
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "issue": mm.ISSUE,
        "generator": "scripts/misc/tooling/build_series_id_map.py",
        "source": mm.MAP_PATH,
        "note": (
            "Pre-move dashboard series ids (legacy file grammar) -> the v3 ids they publish "
            "under, with the releases each held; sources attaches pre-move points outside the "
            "mirrored tree. Generated; do not edit."
        ),
        "counts": {
            "series": len(series),
            "remapped_to_v3": sum(
                not v["id"].startswith(bd.LEGACY_PREFIX) for v in series.values()
            ),
            "kept_legacy": sum(v["id"].startswith(bd.LEGACY_PREFIX) for v in series.values()),
            "published_series": sum(bool(v["releases"]) for v in series.values()),
            "points": sum(len(v["releases"]) for v in series.values()),
            "sources": len(sources),
        },
        "series": series,
        "sources": dict(sorted(sources.items())),
        "cells": runtime_cells(root),
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the map is stale")
    args = parser.parse_args(argv)
    root = bd._profiling_root()
    text = render(build(root))
    path = root / bd.SERIES_ID_MAP
    if args.check:
        if not path.is_file() or path.read_text() != text:
            print(f"{bd.SERIES_ID_MAP} is stale: run build_series_id_map.py")
            return 1
        print(f"{bd.SERIES_ID_MAP} up to date")
        return 0
    path.write_text(text)
    print(f"wrote {bd.SERIES_ID_MAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
