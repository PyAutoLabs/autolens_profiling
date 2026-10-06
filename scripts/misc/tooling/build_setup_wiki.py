"""Generate exact-setup wiki navigation from the exported v2 catalogue (stdlib)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

MARKER = "<!-- generated: build_setup_wiki.py; do not edit -->"


def cell(value):
    return str(value).replace("|", "&#124;").replace("\n", " ")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Unsafe setup navigation path component")
    return value


def page_path(setup):
    component(setup["dataset"])
    component(setup["model"])
    return f"wiki/setups/{setup['dataset']}/{setup['model']}/{hashlib.sha256(setup['id'].encode()).hexdigest()[:20]}.md"


def link(page, target, label):
    return f"[{cell(label)}]({os.path.relpath(target, Path(page).parent)})"


def load_catalogue(root):
    index = json.loads((root / "dashboard/catalogue.json").read_text())
    shards = {}
    for ref in index["evidence_shards"]:
        path = Path(ref["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe evidence shard path")
        raw = (root / "dashboard" / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError(f"Evidence shard checksum mismatch: {path}")
        doc = json.loads(raw)
        if len(doc["records"]) != ref["records"] or [s["id"] for s in doc["setups"]] != [
            ref["setup_id"]
        ]:
            raise ValueError(f"Evidence shard identity/count mismatch: {path}")
        if any(r["setup_id"] != ref["setup_id"] for r in doc["records"]):
            raise ValueError("Evidence shard record setup mismatch")
        shards[ref["setup_id"]] = doc
    return index, shards


def render_outputs(root, index=None, shards=None):
    if index is None:
        index, shards = load_catalogue(root)
    bindings = json.loads((root / "catalogue/wiki_bindings.json").read_text())
    known = {s["id"] for s in index["setups"]}
    campaigns = defaultdict(list)
    for binding in bindings["campaigns"]:
        if not (root / binding["path"]).is_file() or not binding["path"].startswith(
            "wiki/campaigns/"
        ):
            raise ValueError("Missing/invalid campaign binding page")
        for sid in binding["setup_ids"]:
            if sid not in known:
                raise ValueError(f"Unknown campaign setup: {sid}")
            campaigns[sid].append(binding["path"])
    outputs = {}
    groups = defaultdict(list)
    selected = {s["record_id"] for s in index["selections"] if s.get("record_id")}
    for setup in sorted(index["setups"], key=lambda s: s["id"]):
        sid = setup["id"]
        page = page_path(setup)
        groups[(setup["dataset"], setup["model"])].append(setup)
        doc = shards.get(sid)
        records = doc["records"] if doc else []
        if doc and doc["setups"][0] != setup:
            raise ValueError("Index/shard setup mismatch")
        lines = [
            MARKER,
            f"# {cell(setup['label'])}",
            "",
            link(page, f"wiki/setups/{setup['dataset']}/{setup['model']}/index.md", "Model index"),
            "",
            f"Exact setup ID: `{sid}`.",
            "",
            "Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.",
            "",
            "## Configuration",
            "",
            f"Identity limitations: {cell(setup.get('identity_limitations', 'None recorded.'))}",
            "",
            "| Setting | Recorded value | Unit | Unknown reason |",
            "|---|---|---|---|",
        ]
        for key, setting in sorted(setup["configuration"].items()):
            lines.append(
                f"| {cell(key)} | {cell(encoded(setting['value']))} | {cell(setting.get('unit', ''))} | {cell(setting.get('reason', ''))} |"
            )
        lines += ["", "## Evidence", ""]
        ev = setup.get("evidence")
        if ev:
            lines += [
                link(page, ev["path"], "Original setup artifact")
                + f"; JSON pointer: `{ev.get('fragment') or '(document root)'}`.",
                "",
            ]
        if not records:
            lines += ["No measured evidence for this setup. Planned cells remain not measured.", ""]
        lines += ["### Selected references", ""]
        references = [r for r in records if r["id"] in selected]
        lines += [
            f"- `{r['id']}`: {r['axis']} / {r['metric']} = {cell(encoded(r['measurement']))} {r['unit']} ({r['validation']['status']})."
            for r in references
        ] or ["No selected reference for this exact setup; archive evidence is available below."]
        lines += [""]
        for axis in sorted({r["axis"] for r in records}):
            lines += [
                f"### {axis}",
                "",
                f"<details><summary>{sum(r['axis'] == axis for r in records)} recorded {axis} measurements</summary>",
                "",
                "| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in sorted((r for r in records if r["axis"] == axis), key=lambda r: r["id"]):
                identity = r["identity"]
                lines.append(
                    f"| `{r['id']}` | {cell(r['metric'])} | {cell(encoded(r['measurement']))} | {r['unit']} | {cell(identity['device'])} / {cell(identity['precision'])} | {'selected reference' if r['id'] in selected else 'archive support'}; {r['validation']['status']} | {link(page, r['evidence']['path'], 'artifact')} `{r['evidence'].get('fragment') or '(document root)'}` |"
                )
            lines += ["", "</details>", ""]
        # Deduplicate full method/identity blocks, retaining revisions and all unknowns.
        contexts = {
            encoded(
                {
                    "identity": r["identity"],
                    "method": {k: v for k, v in r["method"].items() if k != "id"},
                    "provenance": r["provenance"],
                    "validation": r["validation"],
                }
            )
            for r in records
        }
        for context in sorted(contexts):
            lines += [
                "<details><summary>Recorded hardware, software, method and limitations</summary>",
                "",
                "```json",
                json.dumps(json.loads(context), indent=2, sort_keys=True),
                "```",
                "",
                "</details>",
                "",
            ]
        for kind in ("hazards", "recommendations"):
            lines += [f"## {kind.title()}", ""]
            applicable = [r for r in index[kind] if sid in r["applies_to"]["setup_ids"]]
            if not applicable:
                lines += [
                    "No explicitly bound applicable evidence; this does not establish absence of hazards or validate a setting.",
                    "",
                ]
            for item in applicable:
                lines += [
                    link(
                        page, "dashboard/catalogue.json", "Full versioned applicability and support"
                    ),
                    "",
                    f"### {cell(item['title'])}",
                    "",
                    cell(item["description"]),
                    "",
                    "```json",
                    json.dumps(
                        {
                            "id": item["id"],
                            "validation": item.get("validation", item.get("status")),
                            "limitations": item["applies_to"]["limitations"],
                            "constraints": {
                                k: v
                                for k, v in item["applies_to"]["constraints"].items()
                                if k not in ("measured_software_by_record", "hardware_by_record")
                            },
                            "supporting_record_ids_here": [
                                r
                                for r in item.get("record_ids", [])
                                if r in {record["id"] for record in records}
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                    "",
                ]
                lines += [
                    link(page, e["path"], "Supporting evidence")
                    + f"; pointer `{e.get('fragment') or '(document root)'}`."
                    for e in item["evidence"]
                    if e["path"].endswith(".md")
                    or e["path"] == setup.get("evidence", {}).get("path")
                ]
                lines += [""]
        lines += [
            "## Scripts",
            "",
            "Navigation links identify the model family; they do not reconstruct historical settings or promise an executable replay.",
            "",
        ]
        routes = [
            r
            for r in index["script_routes"]["routes"]
            if r["dataset"] == setup["dataset"] and r["model"] == setup["model"]
        ]
        lines += [f"- {link(page, r['path'], r['measurement'])}" for r in routes] or [
            "No canonical script explicitly registered for this model."
        ]
        lines += ["", "## Campaigns", ""]
        lines += [f"- {link(page, p, Path(p).stem)}" for p in sorted(campaigns[sid])] or [
            "No campaign explicitly bound; campaign applicability is unknown."
        ]
        outputs[page] = "\n".join(lines) + "\n"
    datasets = defaultdict(list)
    for (dataset, model), setups in sorted(groups.items()):
        page = f"wiki/setups/{dataset}/{model}/index.md"
        datasets[dataset].append(model)
        lines = [
            MARKER,
            f"# {dataset} / {model}",
            "",
            link(page, f"wiki/setups/{dataset}/index.md", "Dataset index"),
            "",
            "Each row is an isolated exact configuration, not a combined run. Select the instrument, then the recorded configuration and evidence source.",
            "",
        ]
        for instrument in sorted({s.get("instrument") or "unspecified" for s in setups}):
            lines += [
                f"## {instrument}",
                "",
                "| Exact configuration | Evidence source | Records |",
                "|---|---|---|",
            ]
            for s in setups:
                if (s.get("instrument") or "unspecified") == instrument:
                    lines.append(
                        f"| {link(page, page_path(s), s['configuration_id'])} | {cell(s.get('evidence', {}).get('path', 'planned; no evidence'))} | {len(shards.get(s['id'], {}).get('records', []))} |"
                    )
            lines += [""]
        outputs[page] = "\n".join(lines) + "\n"
    for dataset, models in sorted(datasets.items()):
        page = f"wiki/setups/{dataset}/index.md"
        outputs[page] = (
            "\n".join(
                [MARKER, f"# {dataset}", "", link(page, "wiki/setups/index.md", "Setup index"), ""]
                + [
                    f"- {link(page, f'wiki/setups/{dataset}/{model}/index.md', model)}"
                    for model in models
                ]
            )
            + "\n"
        )
    page = "wiki/setups/index.md"
    outputs[page] = (
        "\n".join(
            [
                MARKER,
                "# Exact setup evidence",
                "",
                link(page, "wiki/index.md", "Campaign journals"),
                "",
                "Choose dataset → model → instrument → exact configuration. Imported evidence is unreviewed. Missing metadata stays unknown; archive rows do not fill baseline plans.",
                "",
            ]
            + [f"- {link(page, f'wiki/setups/{d}/index.md', d)}" for d in sorted(datasets)]
        )
        + "\n"
    )
    allowed = root.resolve() / "wiki/setups"
    for name in outputs:
        target = root / name
        if not target.resolve().is_relative_to(allowed):
            raise ValueError("Setup wiki destination escapes wiki/setups")
        if target.is_file() and not target.read_text().startswith(MARKER):
            raise ValueError(f"Refusing to overwrite non-generated setup page: {name}")
    return outputs


def obsolete_pages(root, outputs):
    return [
        p
        for p in (root / "wiki/setups").rglob("*.md")
        if str(p.relative_to(root)) not in outputs
        and p.read_text().startswith(MARKER)
        and (p.name == "index.md" or re.fullmatch(r"[0-9a-f]{20}\.md", p.name))
    ]


def check(root):
    try:
        outputs = render_outputs(root)
        failures = [
            f"(d) stale generated setup wiki: {name}"
            for name, content in outputs.items()
            if not (root / name).is_file() or (root / name).read_text() != content
        ]
        failures += [
            f"(d) obsolete generated setup wiki: {p.relative_to(root)}"
            for p in obsolete_pages(root, outputs)
        ]
        return failures
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return [f"(d) invalid setup wiki catalogue/bindings: {exc}"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    if args.check:
        failures = check(args.root)
        print("\n".join(failures) if failures else "setup wiki: current")
        return bool(failures)
    outputs = render_outputs(args.root)
    for path in obsolete_pages(args.root, outputs):
        path.unlink()
    for name, content in outputs.items():
        path = args.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f"setup wiki: wrote {len(outputs)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
