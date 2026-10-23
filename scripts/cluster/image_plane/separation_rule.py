"""
Cluster PointSolver: does the coarsest safe starting scale track an image separation?
====================================================================================

autolens_profiling#439 (follow-up to the phase-1 robust-settings sweep, #436). For every
judged draw-source of one or more ``robust_settings_sweep.py`` JSONs this joins the
oracle's considered roots (``|mu|`` above the solver's threshold band, the judge's set)
with the draw's lens mass centres and records

- the smallest image-to-image separation and the smallest image-to-member-core
  separation (main lenses + scaling members; the NFW host is excluded, it shares the
  BCG's centre), per draw-source and per root;
- the per-scale verdict of the production-extent rows at the production precision
  (``block == "geometry"``, ``extent == "production"``, JIT, library
  ``MAX_CONTAINING_SIZE``): a scale *covers* a draw-source when no considered root is
  missed or duplicated (spurious candidates are counted separately);
- ``coarsest_covering`` (largest tested scale that covers) and ``coarsest_safe``
  (largest tested scale ``s`` such that every tested scale ``<= s`` covers). Censoring:
  ``lower`` = safe at every tested scale (the true value is at least the largest tested
  scale); ``upper`` = the finest tested scale already misses.

Rule test. A rule "safe when scale <= separation / c" is violated by any covered-failure
at a scale ``s <= separation / c``; the smallest ``c`` that the data allow is
``c_required = max(separation / s_fail)`` over every failing (draw-source, scale), per
separation kind and per root. A rule is stated only if it is not contradicted by a
failure at a scale far below the separation (see ``rule`` in the JSON).

Several JSONs may describe the same draw (e.g. the phase-1 RAL JSON and a laptop
``--draw-ids`` re-run at finer scales); per-scale verdicts are merged by
``(dataset, draw_id, source)`` and any disagreement at a shared scale is listed under
``conflicts``.

A2744 (``--a2744-log``): the pilot job 412573 wrote no JSON (it died on LLVM memory), so
only its progress lines' row-level counts exist -- no per-root positions, so no
separations. Its rows enter the table and the right-hand panel (missed fraction vs
scale) only, at the four phase-1 scales (0.5-2.0"), so any "coarsest safe" from them is
bounded by that grid.

Output: ``<stem>.json``, ``<stem>.md`` and ``<stem>.png`` (left: coarsest safe scale vs
the smaller of the two separations, one marker per draw-source, datasets and censoring
distinguished by legend symbols only; right: missed-root fraction vs scale per dataset).

Run::

    python scripts/cluster/image_plane/separation_rule.py \\
        results/breakdown/cluster/robust_settings_sweep_viz_cluster_laptop_cpu_fp64.json \\
        results/breakdown/cluster/robust_settings_sweep_simple_hpc_ral_cpu_fp64.json \\
        results/breakdown/cluster/robust_settings_sweep_simple_draws53_73_81_laptop_cpu_fp64.json \\
        --a2744-log results/logs/cluster_pointsolver/<pilot>.out
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _judge import MAGNIFICATION_THRESHOLD, MU_AMBIGUITY_BAND  # noqa: E402

CORE_KINDS = ("main", "scaling")
LIBRARY_MCS = 20

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on synthetic tables)
# ---------------------------------------------------------------------------


def considered(roots) -> list:
    """The judge's considered set: accepted roots with |mu| above the ambiguity band."""
    hi = MAGNIFICATION_THRESHOLD * (1 + MU_AMBIGUITY_BAND)
    return [r for r in roots if r.get("accepted", True) and abs(r["magnification"]) > hi]


def nearest(points, others, *, exclude_self=False) -> tuple[np.ndarray, np.ndarray]:
    """Per point: distance to and index of its nearest ``others`` entry."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    o = np.asarray(others, dtype=float).reshape(-1, 2)
    if not len(p) or not len(o):
        return np.full(len(p), np.inf), np.full(len(p), -1)
    d = np.hypot(p[:, None, 0] - o[None, :, 0], p[:, None, 1] - o[None, :, 1])
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    idx = np.argmin(d, axis=1)
    return d[np.arange(len(p)), idx], idx


def scale_summary(status_by_scale: dict) -> dict:
    """``{scale: True (covers) / False}`` -> coarsest covering / safe, censoring, s_fail."""
    scales = sorted(status_by_scale)
    if not scales:
        return {
            "coarsest_covering": None,
            "coarsest_safe": None,
            "censoring": None,
            "finest_failing": None,
            "monotone": None,
        }
    covering = [s for s in scales if status_by_scale[s]]
    failing = [s for s in scales if not status_by_scale[s]]
    safe = None
    for s in scales:  # ascending
        if status_by_scale[s]:
            safe = s
        else:
            break
    censoring = "lower" if safe == scales[-1] else ("upper" if safe is None else None)
    monotone = not failing or not covering or max(covering) < min(failing)
    return {
        "coarsest_covering": max(covering) if covering else None,
        "coarsest_safe": safe,
        "censoring": censoring,
        "finest_failing": min(failing) if failing else None,
        "monotone": bool(monotone),
        "tested_scales": scales,
    }


def required_c(records, sep_key: str) -> dict:
    """Smallest ``c`` with "scale <= sep / c is safe" uncontradicted, plus the binding case.

    ``records``: dicts with ``sep_key`` and ``failing_scales`` (list). A failure at scale
    ``s`` contradicts every ``c <= sep / s``.
    """
    worst, binding, n_fail = 0.0, None, 0
    for r in records:
        sep = r.get(sep_key)
        if sep is None or not np.isfinite(sep):
            continue
        for s in r.get("failing_scales", []):
            n_fail += 1
            ratio = sep / s
            if ratio > worst:
                worst, binding = ratio, {k: r.get(k) for k in ("id", sep_key)} | {"scale": s}
    return {"c_required": worst if n_fail else None, "binding": binding, "n_failures": n_fail}


A2744_ROW = re.compile(
    r"row (?P<name>\S+)\s+(?P<status>admissible|invalid|diagnostic|unusable)\s+cov "
    r"(?P<cov>\d+)/(?P<n>\d+) \[prior (?P<pc>\d+)/(?P<pn>\d+)\] \[stress (?P<sc>\d+)/(?P<sn>\d+)\] "
    r"miss (?P<miss>\d+) dup (?P<dup>\d+) spur (?P<spur>\d+).*?maxerr (?P<maxerr>\S+) "
    r"\[rss (?P<rss>\d+) MB, peak (?P<peak>\d+) MB"
)
A2744_INVALID = re.compile(
    r"row (?P<name>\S+): INVALID \(route error (?P<err>[^:]+): (?P<msg>.*?)\) \[rss (?P<rss>\d+) MB"
)
ROW_NAME = re.compile(r"(?P<extent>production|bbox|tight)_s(?P<scale>[\d.]+)_p(?P<prec>[\d.]+)")


def parse_row_name(name: str, control=(1.0, 0.001)) -> dict:
    if name == "control":
        return {"extent": "production", "scale": control[0], "precision": control[1], "mcs": None}
    base, _, mcs = name.partition("_mcs")
    if base == "control":
        return {
            "extent": "production",
            "scale": control[0],
            "precision": control[1],
            "mcs": int(mcs) if mcs else None,
        }
    m = ROW_NAME.fullmatch(base)
    if not m:
        return {"extent": None, "scale": None, "precision": None, "mcs": None}
    return {
        "extent": m["extent"],
        "scale": float(m["scale"]),
        "precision": float(m["prec"]),
        "mcs": int(mcs) if mcs else None,
    }


def parse_a2744_log(text: str) -> list[dict]:
    """Row lines of a sweep progress log (job 412573) -> row dicts (no per-root data)."""
    rows, seen = [], set()
    for line in text.splitlines():
        m = A2744_ROW.search(line)
        if m:
            info = parse_row_name(m["name"])
            rows.append(
                {
                    "config": m["name"],
                    **info,
                    "status": m["status"],
                    "coverage": f"{m['cov']}/{m['n']}",
                    "prior": f"{m['pc']}/{m['pn']}",
                    "stress": f"{m['sc']}/{m['sn']}",
                    "n_considered": int(m["n"]),
                    "n_missed": int(m["miss"]),
                    "n_missed_prior": int(m["pn"]) - int(m["pc"]),
                    "n_missed_stress": int(m["sn"]) - int(m["sc"]),
                    "n_duplicated": int(m["dup"]),
                    "n_spurious": int(m["spur"]),
                    "max_position_error": float(m["maxerr"]),
                    "rss_mb": int(m["rss"]),
                    "peak_rss_mb": int(m["peak"]),
                }
            )
            continue
        m = A2744_INVALID.search(line)
        if m and m["name"] not in seen:
            seen.add(m["name"])
            rows.append(
                {
                    "config": m["name"],
                    **parse_row_name(m["name"]),
                    "status": "invalid",
                    "error": f"{m['err']}: {m['msg']}",
                    "rss_mb": int(m["rss"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Sweep JSON -> per draw-source records
# ---------------------------------------------------------------------------


def geometry_rows(sweep: dict) -> dict:
    """{scale: row} for the production-extent JIT rows at the production precision."""
    prec = float(sweep["production_solver"]["pixel_scale_precision"])
    out = {}
    for row in sweep["rows"].values():
        cfg = row["config"]
        if (
            cfg["block"] == "geometry"
            and cfg["extent"] == "production"
            and cfg["backend"] == "jit"
            and cfg["max_containing_size"] == LIBRARY_MCS
            and math.isclose(cfg["precision"], prec)
            and row.get("per_draw")
        ):
            out[float(cfg["scale"])] = row
    return out


def records_from_sweep(sweep: dict, centres_for, label: str) -> list[dict]:
    """One record per judged draw-source; ``centres_for(vector) -> [{kind, name, centre}]``."""
    rows = geometry_rows(sweep)
    draws = sweep["draws"]
    ids = draws.get("draw_ids") or list(range(draws["n_judged"]))
    sets = draws["set_per_draw"]
    out = []
    for local, did in enumerate(ids):
        cores = [c for c in centres_for(draws["vectors"][local]) if c["kind"] in CORE_KINDS]
        core_xy = [c["centre"] for c in cores]
        osrc = sweep["oracle"]["per_draw"][local]["sources"]
        for k, src in enumerate(osrc):
            roots = considered(src["roots"])
            if not roots:
                continue
            pos = [r["position"] for r in roots]
            d_img, _ = nearest(pos, pos, exclude_self=True)
            d_core, i_core = nearest(pos, core_xy)
            per_scale, per_root_fail, spurious = {}, [[] for _ in roots], {}
            for s, row in rows.items():
                entry = row["per_draw"][local]["sources"][k]
                if entry is None:
                    continue
                verdicts = {tuple(np.round(r["position"], 9)): r["verdict"] for r in entry["roots"]}
                ok = True
                for j, p in enumerate(pos):
                    v = verdicts.get(tuple(np.round(p, 9)))
                    if v != "covered":
                        ok = False
                        per_root_fail[j].append(s)
                per_scale[s] = ok
                spurious[s] = int(entry["n_spurious"])
            out.append(
                {
                    "id": f"{sweep['dataset']}/{did}/{k}",
                    "group": label,
                    "dataset": sweep["dataset"],
                    "set": sets[local],
                    "draw_id": int(did),
                    "source": k,
                    "n_roots": len(roots),
                    "min_image_image_sep": float(d_img.min()) if len(roots) > 1 else None,
                    "min_image_core_sep": float(d_core.min()),
                    "nearest_core": cores[int(i_core[np.argmin(d_core)])]["name"],
                    "per_scale_covers": {str(s): v for s, v in sorted(per_scale.items())},
                    "per_scale_spurious": {str(s): v for s, v in sorted(spurious.items())},
                    "roots": [
                        {
                            "position": p,
                            "magnification": r["magnification"],
                            "nearest_image_sep": float(d_img[j]) if len(roots) > 1 else None,
                            "nearest_core_sep": float(d_core[j]),
                            "nearest_core": cores[int(i_core[j])]["name"],
                            "failing_scales": sorted(per_root_fail[j]),
                        }
                        for j, (p, r) in enumerate(zip(pos, roots))
                    ],
                    "sources_files": [sweep.get("_file")],
                }
            )
    return out


def merge(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge records of the same (dataset, draw_id, source) across JSONs; list conflicts."""
    by_id, conflicts = {}, []
    for r in records:
        prev = by_id.get(r["id"])
        if prev is None:
            by_id[r["id"]] = r
            continue
        more_scales = len(r["per_scale_covers"]) > len(prev["per_scale_covers"])
        for s, v in r["per_scale_covers"].items():
            if s in prev["per_scale_covers"] and prev["per_scale_covers"][s] != v:
                conflicts.append(
                    {
                        "id": r["id"],
                        "scale": float(s),
                        "kept": prev["per_scale_covers"][s],
                        "other": v,
                        "files": prev["sources_files"] + r["sources_files"],
                    }
                )
            else:
                prev["per_scale_covers"][s] = v
        for s, v in r["per_scale_spurious"].items():
            prev["per_scale_spurious"].setdefault(s, v)
        for rr in r["roots"]:
            d = [np.hypot(*np.subtract(rp["position"], rr["position"])) for rp in prev["roots"]]
            if d and min(d) < 1e-6:
                rp = prev["roots"][int(np.argmin(d))]
                rp["failing_scales"] = sorted(set(rp["failing_scales"]) | set(rr["failing_scales"]))
            else:
                conflicts.append({"id": r["id"], "unmatched_root": rr["position"]})
        prev["sources_files"] += r["sources_files"]
        if more_scales:
            prev["group"] = r["group"]  # the record with more scales labels the merge
    return list(by_id.values()), conflicts


def finalise(record: dict) -> dict:
    covers = {float(s): v for s, v in record["per_scale_covers"].items()}
    record.update(scale_summary(covers))
    record["failing_scales"] = sorted(s for s, v in covers.items() if not v)
    seps = [x for x in (record["min_image_image_sep"], record["min_image_core_sep"]) if x]
    record["min_sep"] = min(seps) if seps else None
    return record


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


def rule_test(records: list[dict]) -> dict:
    root_recs = [
        {
            "id": f"{r['id']}@{np.round(rt['position'], 3).tolist()}",
            "nearest_image_sep": rt["nearest_image_sep"],
            "nearest_core_sep": rt["nearest_core_sep"],
            "min_feature_sep": min(
                x for x in (rt["nearest_image_sep"], rt["nearest_core_sep"]) if x is not None
            ),
            "failing_scales": rt["failing_scales"],
        }
        for r in records
        for rt in r["roots"]
    ]
    out = {
        "per_draw_source": {
            k: required_c(records, k)
            for k in ("min_image_image_sep", "min_image_core_sep", "min_sep")
        },
        "per_root": {
            k: required_c(root_recs, k)
            for k in ("nearest_image_sep", "nearest_core_sep", "min_feature_sep")
        },
        "n_records": len(records),
        "n_records_failing_somewhere": sum(bool(r["failing_scales"]) for r in records),
        "n_non_monotone": sum(r["monotone"] is False for r in records),
        "claimed_rule_c": 1.06,
    }
    cr = out["per_root"]["min_feature_sep"]["c_required"]
    out["claimed_rule_holds_per_root"] = None if cr is None else bool(cr < 1.06)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(records, rule, a2744_rows, conflicts) -> str:
    lines = [
        "# Cluster PointSolver: coarsest safe starting scale vs separation (#439)",
        "",
        "Production extent (workspace grid ±49.5″), precision 0.001, magnification threshold 0.1; "
        "a scale covers a draw-source when no considered oracle root is missed or duplicated. "
        "`safe` = largest tested scale with every finer tested scale covering; `≥` = safe at every "
        "tested scale, `<` = the finest tested scale misses.",
        "",
        "## Draw-sources that miss at some tested scale",
        "",
        "| Group | Set | Draw/src | Images | Min img–img ″ | Min img–core ″ (core) | Missing at | "
        "Coarsest covering | Coarsest safe |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    failing = sorted(
        (r for r in records if r["failing_scales"]), key=lambda r: (r["group"], r["min_sep"] or 0)
    )
    for r in failing:
        safe = r["coarsest_safe"]
        safe_s = (
            f"< {min(r['tested_scales']):g}"
            if r["censoring"] == "upper"
            else (f"≥ {safe:g}" if r["censoring"] == "lower" else f"{safe:g}")
        )
        lines.append(
            f"| {r['group']} | {r['set']} | {r['draw_id']}/{r['source']} | {r['n_roots']} | "
            f"{_fmt(r['min_image_image_sep'])} | {_fmt(r['min_image_core_sep'])} ({r['nearest_core']}) | "
            f"{', '.join(f'{s:g}' for s in r['failing_scales'])} | "
            f"{_fmt(r['coarsest_covering'], 2) if r['coarsest_covering'] else '—'} | {safe_s} |"
        )
    n_ok = {}
    for r in records:
        if not r["failing_scales"]:
            n_ok.setdefault(r["group"], [0, None])
            n_ok[r["group"]][0] += 1
            n_ok[r["group"]][1] = max(r["tested_scales"])
    lines += [
        "",
        "Safe at every tested scale: "
        + "; ".join(f"{g}: {n} draw-sources (≥ {s:g}″)" for g, (n, s) in sorted(n_ok.items()))
        + ".",
    ]
    lines += ["", "## Rule test", ""]
    lines += [
        '`c_required` = the smallest c for which "scale ≤ separation / c is safe" is not '
        "contradicted by any observed miss (a miss at scale s contradicts every c ≤ sep / s).",
        "",
        "| Unit | Separation | c_required | Binding case | Failures |",
        "|---|---|---|---|---|",
    ]
    for unit in ("per_root", "per_draw_source"):
        for k, v in rule[unit].items():
            b = v["binding"] or {}
            lines.append(
                f"| {unit} | {k} | {_fmt(v['c_required'])} | {b.get('id', '—')} at {b.get('scale', '—')}″ | "
                f"{v['n_failures']} |"
            )
    lines += [
        "",
        f"Claimed rule (scale ≲ separation / 1.06) holds per root: {rule['claimed_rule_holds_per_root']}. "
        f"Non-monotone draw-sources (a coarser scale covers, a finer one misses): {rule['n_non_monotone']}.",
    ]
    if conflicts:
        lines += ["", "## Conflicts between JSONs at a shared scale", ""]
        lines += [
            f"- {c['id']} at {c['scale']:g}″: kept {c['kept']}, other {c['other']}"
            for c in conflicts
        ]
    if a2744_rows:
        lines += [
            "",
            "## A2744 pilot rows (job 412573; row-level only, no per-root record)",
            "",
            "| Row | Scale | Prec | Status | Coverage | Prior | Stress | Missed | Max err ″ | RSS MB | Peak MB |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for a in a2744_rows:
            lines.append(
                f"| {a['config']} | {a.get('scale') or '—'} | {a.get('precision') or '—'} | {a['status']} | "
                f"{a.get('coverage', '—')} | {a.get('prior', '—')} | {a.get('stress', '—')} | "
                f"{a.get('n_missed', '—')} | {a.get('max_position_error', '—')} | {a.get('rss_mb', '—')} | "
                f"{a.get('peak_rss_mb', a.get('error', '—'))} |"
            )
    return "\n".join(lines) + "\n"


GROUP_STYLE = {
    "viz_cluster": dict(marker="o", color="#4C72B0"),
    "simple stress 53/73/81 (laptop, 8 scales)": dict(marker="s", color="#C44E52"),
    "simple (RAL phase 1, 4 scales)": dict(marker="^", color="#8C8C8C"),
}


def figure(records, a2744_rows, path, sweeps_missed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13, 5.5), constrained_layout=True)
    seen = set()
    for r in records:
        if r["min_sep"] is None:
            continue
        st = GROUP_STYLE.get(r["group"], dict(marker="D", color="black"))
        cens = r["censoring"]
        y = (
            r["coarsest_safe"]
            if cens is None
            else (max(r["tested_scales"]) if cens == "lower" else 0.5 * min(r["tested_scales"]))
        )
        face = st["color"] if cens is None else "none"
        label = (
            r["group"]
            + {
                None: "",
                "lower": " (safe at every tested scale: lower bound)",
                "upper": " (misses at the finest tested scale: upper bound, drawn at half of it)",
            }[cens]
        )
        ax1.scatter(
            r["min_sep"],
            y,
            marker=st["marker"],
            facecolors=face,
            edgecolors=st["color"],
            s=46,
            label=None if label in seen else label,
            zorder=3,
        )
        seen.add(label)
    xs = np.logspace(-1.3, 1.6, 50)
    ax1.plot(
        xs, xs / 1.06, color="black", lw=1, ls="--", label="claimed rule: scale = separation / 1.06"
    )
    ax1.axhline(1.0, color="#DD8452", lw=1, ls=":", label='workspace starting scale 1.0"')
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel('smallest image-image or image-member-core separation ["]')
    ax1.set_ylabel('coarsest safe starting scale ["]')
    ax1.legend(fontsize=7, loc="lower right")

    for label, pts in sweeps_missed.items():
        st = GROUP_STYLE.get(label, dict(marker="o", color="#4C72B0"))
        s, f = zip(*sorted(pts))
        ax2.plot(s, f, marker=st["marker"], color=st["color"], label=label)
    if a2744_rows:
        pts = sorted(
            (a["scale"], a["n_missed"] / a["n_considered"])
            for a in a2744_rows
            if a.get("extent") == "production" and a.get("precision") == 0.001 and "n_missed" in a
        )
        if pts:
            s, f = zip(*pts)
            ax2.plot(
                s,
                f,
                marker="P",
                color="#55A868",
                label="A2744 pilot 412573 (synthetic prior-median model, 3 draws; scales 0.5-2.0 only)",
            )
    ax2.axvline(1.0, color="#DD8452", lw=1, ls=":", label='workspace starting scale 1.0"')
    ax2.set_xscale("log")
    ax2.set_xlabel('starting scale ["]')
    ax2.set_ylabel("missed fraction of considered oracle roots")
    ax2.legend(fontsize=7)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def missed_fraction_by_scale(sweep: dict) -> list[tuple[float, float]]:
    out = []
    for s, row in geometry_rows(sweep).items():
        v = row["verdict"]
        if v["n_roots_considered"]:
            out.append((s, v["n_missed"] / v["n_roots_considered"]))
    return out


def group_label(sweep: dict) -> str:
    if sweep["dataset"] == "viz_cluster":
        return "viz_cluster"
    if sweep["dataset"] == "simple" and sweep["draws"].get("draw_id_filter"):
        return "simple stress 53/73/81 (laptop, 8 scales)"
    return "simple (RAL phase 1, 4 scales)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("sweeps", nargs="+", type=Path)
    ap.add_argument("--a2744-log", type=Path, default=None)
    ap.add_argument(
        "--out-stem",
        type=Path,
        default=_HERE.parents[2] / "results" / "breakdown" / "cluster" / "separation_rule",
    )
    args = ap.parse_args(argv)

    import _datasets

    cases = {}
    records, missed = [], {}
    for path in args.sweeps:
        sweep = json.loads(path.read_text())
        sweep["_file"] = path.name
        ds = sweep["dataset"]
        if ds not in cases:
            cases[ds] = _datasets.load_case(ds)
        label = group_label(sweep)
        records += records_from_sweep(sweep, cases[ds].mass_centres, label)
        missed[label] = missed_fraction_by_scale(sweep)
        del sweep
    records, conflicts = merge(records)
    records = [finalise(r) for r in records]
    rule = rule_test(records)
    a2744 = parse_a2744_log(args.a2744_log.read_text()) if args.a2744_log else []
    out = {
        "issue": "PyAutoLabs/autolens_profiling#439",
        "inputs": [p.name for p in args.sweeps],
        "a2744_log": args.a2744_log.name if args.a2744_log else None,
        "definitions": {
            "covers": "no considered oracle root missed or duplicated at that scale",
            "considered": f"accepted oracle roots with |mu| > {MAGNIFICATION_THRESHOLD} x (1 + {MU_AMBIGUITY_BAND})",
            "member_cores": list(CORE_KINDS),
            "rows": "production extent, production precision, JIT, library MAX_CONTAINING_SIZE",
        },
        "rule": rule,
        "conflicts": conflicts,
        "a2744_rows": a2744,
        "a2744_note": "job 412573 wrote no JSON: row-level counts only, no per-root positions, so "
        "no separations; scales 0.5-2.0 only",
        "records": records,
    }
    stem = args.out_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1))
    stem.with_suffix(".md").write_text(markdown(records, rule, a2744, conflicts))
    figure(records, a2744, stem.with_suffix(".png"), missed)
    print(stem.with_suffix(".md").read_text())
    return out


if __name__ == "__main__":
    main()
