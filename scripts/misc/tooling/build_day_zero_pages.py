"""Render the day-zero rule pages: one ``wiki/campaigns/<dataset_type>/<variant>/day_zero.md``
per designated likelihood, plus the ``wiki/day_zero.md`` index (stdlib only).

Run from the repo root::

    python scripts/misc/tooling/build_day_zero_pages.py           # write
    python scripts/misc/tooling/build_day_zero_pages.py --check   # exit 1 on drift

Inputs (nothing else is read, nothing is measured or ranked here):

- ``catalogue/pulse_designations.json``: the PyAutoPulse designation records (rule sentence in
  dataset-property terms, optimal, secondary, record URL);
- ``catalogue/designation_holds.json``: groups the human held provisional;
- ``catalogue/day_zero.json``: the day-zero tag;
- ``dashboard/catalogue.json``: the stamped records (evidence) and what was measured (the
  untested branches are what the rule extrapolates to beyond it).

Likelihood lifecycle phase 3, day-zero write 3/4 (autolens_profiling#501). The pages are
generated; edit the inputs or this renderer, never the pages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DESIGNATIONS = "catalogue/pulse_designations.json"
HOLDS = "catalogue/designation_holds.json"
DAY_ZERO = "catalogue/day_zero.json"
CATALOGUE = "dashboard/catalogue.json"
INDEX = "wiki/day_zero.md"
SNAPSHOT_URL = "https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json"
#: Phase-6 gradient stamps are frozen apart (``freeze --measurement gradient``, #512).
GRADIENT_SNAPSHOT_URL = (
    "https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero_gradient/lens.json"
)
HARDWARE = {
    "cpu_many_core": ("hpc_ral_cpu", "Many-core CPU node, no GPU"),
    "gpu_a100": ("hpc_a100", "A100 GPU node"),
}
HOST_HARDWARE = {host: hw for hw, (host, _label) in HARDWARE.items()}
STAMP = "<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->"


def page_path(likelihood: str) -> str:
    return f"wiki/campaigns/{likelihood}/day_zero.md"


def _load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text())


def _sig(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def _duration(seconds) -> str:
    if not seconds:
        return "—"
    return f"{_sig(seconds * 1e3)} ms" if seconds < 1 else f"{_sig(seconds)} s"


def _rel(from_page: str, target: str) -> str:
    return os.path.relpath(target, os.path.dirname(from_page)).replace(os.sep, "/")


def _setups(entry: dict) -> tuple[list[str], list[str]]:
    optimal = entry["optimal"]
    return (optimal if isinstance(optimal, list) else [optimal]), list(entry["secondary"])


def _untested_clause(rule: str) -> str | None:
    marker = "Untested:"
    return rule.split(marker, 1)[1].strip() if marker in rule else None


def _rule_section(likelihood, designations, holds) -> list[str]:
    lines = ["## Rule", ""]
    for hw, (host, label) in HARDWARE.items():
        entry = designations.get((likelihood, hw))
        if entry is not None:
            optimal, secondary = _setups(entry)
            fallback = ", ".join(f"`{s}`" for s in secondary) or "none"
            lines += [
                f"- **{label}** (`{hw}`): {entry['rule']}",
                f"  Optimal `{optimal[0]}`; secondary {fallback}. "
                f"Pulse record [`{entry['id']}`]({entry['url']}) ({entry['date']}).",
            ]
            continue
        hold = holds.get((likelihood, host))
        if hold is not None:
            optimal, secondary = _setups(hold)
            lines += [
                f"- **{label}** (`{hw}`): **held, not designated.** Provisional (lifecycle "
                f"report section 3.3): {', '.join(f'`{s}`' for s in optimal)}; fallback "
                f"{', '.join(f'`{s}`' for s in secondary) or 'none'}. {hold['decision']}",
            ]
            continue
        lines.append(f"- **{label}** (`{hw}`): no designation and no hold.")
    return lines + [""]


def _evidence_section(page, likelihood, records) -> list[str]:
    rows = []
    for r in records:
        if r["measurement"] != "runtime" or r["day_zero"] is None:
            continue
        hosts = (r.get("projected_node_throughput") or {}).get("hosts") or []
        node = next((h for h in hosts if h.get("host_class") == r["host_class"]), None)
        per_call = (r.get("projected_node_throughput") or {}).get("per_call_s")
        rows.append(
            (
                r["host_class"],
                r["instrument"],
                {"optimal": 0, "secondary": 1}.get(r["class"], 2),
                r["setup_id"],
                "| {inst} | `{host}` | `{setup}` | {cls}{dec} | {per_call} | {calls} | [`{rid}`]({path}) |".format(
                    inst=r["instrument"],
                    host=r["host_class"],
                    setup=r["setup_id"],
                    cls=r["class"],
                    dec="" if r["decision_id"] else " (held)",
                    per_call=_duration(per_call),
                    calls=_sig(node["calls_per_s"], 3) if node and node.get("calls_per_s") else "—",
                    rid=r["id"],
                    path=_rel(page, r["evidence"]["path"]),
                ),
            )
        )
    lines = [
        "## Day-zero evidence",
        "",
        "Stamped current runtime records (each cell's parity witness is a `match`; the cell's "
        "breakdown, compile, memory and parity records carry the same stamp). Per call is the "
        "batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per "
        "node with parallel efficiency 1.0 (declared, unmeasured).",
        "",
        "| instrument | host | setup | class | per call | projected calls/s per node | record |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [row[-1] for row in sorted(rows)] or ["| — | — | — | — | — | — | — |"]
    return lines + [""]


def _untested_section(likelihood, designations, holds, records, all_records) -> list[str]:
    items = []
    for hw in HARDWARE:
        entry = designations.get((likelihood, hw))
        clause = entry and _untested_clause(entry["rule"])
        if clause:
            items.append(f"Named in the `{hw}` rule: {clause}")
    for hw, (host, label) in HARDWARE.items():
        if (likelihood, host) in holds:
            items.append(
                f"`{hw}`: the setups do not separate; the cell stays provisional until a "
                "designation is recorded."
            )
    current = [r for r in records if r["status"] == "current"]
    instruments = sorted({r["instrument"] for r in current if r["regime"] == "standard"})
    items.append(
        "Instruments: measured on "
        + ", ".join(f"`{i}`" for i in instruments)
        + " only; the rule extrapolates to any other instrument, data size or "
        "visibility/pixel count."
    )
    sizes = sorted({r["regime"] for r in current if r["regime"] != "standard"})
    sizes_text = (
        "measured but not designated: "
        + ", ".join(f"`{s}`" for s in sizes)
        + " (rows inherit the role as secondary, no decision); "
        if sizes
        else ""
    )
    items.append(
        f"Sizes: only the `standard` regime is designated; {sizes_text}other sizes are unmeasured."
    )
    items.append(
        "Hosts: laptop and local GPUs have no designation (local rows inherit the role, no "
        "decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured."
    )
    designated = set()
    for hw in HARDWARE:
        entry = designations.get((likelihood, hw))
        if entry is not None:
            optimal, secondary = _setups(entry)
            designated |= set(optimal) | set(secondary)
    rivals = sorted(
        {
            r["setup_id"]
            for r in current
            if r["class"] == "experimental" and r["setup"]["precision"] != "mixed"
        }
    )
    if rivals:
        items.append(
            "Measured on the designated hosts but not named by the rule (class "
            "`experimental`): " + ", ".join(f"`{s}`" for s in rivals) + "."
        )
    mixed = sorted({r["setup_id"] for r in all_records if r["setup"]["precision"] == "mixed"})
    items.append(
        "Mixed precision: "
        + (
            "measured (" + ", ".join(f"`{s}`" for s in mixed) + ") and experimental"
            if mixed
            else "unmeasured; experimental"
        )
        + " at day zero."
    )
    items.append(
        "Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); "
        "phase 4 measures it, and a measured value that reorders the setups supersedes the "
        "designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled."
    )
    return ["## Untested branches", ""] + [f"- {item}" for item in items] + [""]


def _revisit_section(likelihood, designations) -> list[str]:
    links = [
        f"[`{entry['id']}`]({entry['url']}#revisit-conditions)"
        for hw in HARDWARE
        if (entry := designations.get((likelihood, hw))) is not None
    ]
    return [
        "## Revisit",
        "",
        "Each designation's revisit conditions live in its Pulse record: "
        + (", ".join(links) or "none")
        + ".",
        "",
    ]


def render(root: Path = ROOT) -> dict[str, str]:
    """{repo-relative path: text} for every page and the index."""
    mirror = _load(root, DESIGNATIONS)["designations"]
    designations = {(r["likelihood"], r["hardware_class"]): r for r in mirror}
    holds = {(h["likelihood"], h["host_class"]): h for h in _load(root, HOLDS)["holds"]}
    tag = _load(root, DAY_ZERO)["tag"]
    records = _load(root, CATALOGUE)["records"]
    by_likelihood = {}
    for r in records:
        key = f"{r['likelihood']['dataset_type']}/{r['likelihood']['variant']}"
        by_likelihood.setdefault(key, []).append(r)
    likelihoods = sorted({lk for lk, _hw in designations} | {lk for lk, _h in holds})
    out, index_rows = {}, []
    for likelihood in likelihoods:
        page = page_path(likelihood)
        mine = by_likelihood.get(likelihood, [])
        reference = [r for r in mine if r["host_class"] in HOST_HARDWARE]
        stamped = sum(1 for r in mine if r["day_zero"] is not None)
        gradient = sum(
            1 for r in mine if r["day_zero"] is not None and r.get("measurement") == "gradient"
        )
        frozen = f"frozen in the [PyAutoPulse day_zero snapshot]({SNAPSHOT_URL})"
        if gradient:
            frozen = (
                f"{stamped - gradient} {frozen} and {gradient} gradient record(s) in the "
                f"[day_zero_gradient snapshot]({GRADIENT_SNAPSHOT_URL})"
            )
        lines = [
            STAMP,
            f"# Day zero: {likelihood}",
            "",
            f"Baseline tag `{tag}` ([day_zero.json]({_rel(page, DAY_ZERO)})); "
            f"{stamped} stamped records, {frozen}. "
            f"[All likelihoods]({_rel(page, INDEX)}).",
            "",
        ]
        lines += _rule_section(likelihood, designations, holds)
        lines += _evidence_section(page, likelihood, mine)
        lines += _untested_section(likelihood, designations, holds, reference, mine)
        lines += _revisit_section(likelihood, designations)
        out[page] = "\n".join(lines).rstrip("\n") + "\n"
        cells = []
        for hw, (host, _label) in HARDWARE.items():
            entry = designations.get((likelihood, hw))
            if entry is not None:
                cells.append(f"[`{_setups(entry)[0][0]}`]({entry['url']})")
            elif (likelihood, host) in holds:
                cells.append("held (provisional)")
            else:
                cells.append("—")
        index_rows.append(
            f"| [{likelihood}]({_rel(INDEX, page)}) | {cells[0]} | {cells[1]} | {stamped} |"
        )
    out[INDEX] = (
        "\n".join(
            [
                STAMP,
                "# Day zero: the optimal setup per likelihood",
                "",
                f"Baseline tag `{tag}` (likelihood lifecycle phase 3). Each page states the "
                "rule in dataset-property terms per hardware class, links its PyAutoPulse "
                "designation record(s), lists the stamped evidence and names the untested "
                f"branches. The stamped records are frozen in the [PyAutoPulse day_zero "
                f"snapshot]({SNAPSHOT_URL}); the phase-6 gradient stamps in the "
                f"[day_zero_gradient snapshot]({GRADIENT_SNAPSHOT_URL}).",
                "",
                "| likelihood | many-core CPU (`cpu_many_core`) | A100 (`gpu_a100`) | stamped records |",
                "|---|---|---|---|",
                *index_rows,
            ]
        )
        + "\n"
    )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    outputs = render(root)
    stale = [
        p
        for p, text in outputs.items()
        if not (root / p).is_file() or (root / p).read_text() != text
    ]
    if args.check:
        if stale:
            print(
                f"day_zero pages: STALE ({len(stale)}, e.g. {stale[0]}); run build_day_zero_pages.py"
            )
            return 1
        print(f"day_zero pages: current ({len(outputs) - 1} pages + index)")
        return 0
    for p in stale:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(outputs[p])
    print(f"day_zero pages: wrote {len(stale)} of {len(outputs)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
