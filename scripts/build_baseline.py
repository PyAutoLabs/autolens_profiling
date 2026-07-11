"""
build_baseline.py — snapshot campaign results into a named, frozen baseline.

A **baseline** (convention: `results/notes/design_lock_in.md`) is a named
snapshot of campaign results under `results/baselines/<Name>/`:

- the `comparison.json` per swept cell, mirroring the `results/runtime/`
  layout, plus
- a rendered `<Name>.md` — every headline number on one browsable page.

The dashboard (`build_readme.py`) grows a baseline column automatically once
the directory exists. Baselines are **append-only after the campaign
closes**; while a campaign is accumulating, re-running this script refreshes
the snapshot from the current `results/runtime/` tree.

Run from the repo root:

    python scripts/build_baseline.py --name PreOptimizationTimes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_readme import (  # noqa: E402
    BASELINES_ROOT,
    RUNTIME_ROOT,
    _config_headline_seconds,
    _config_vmap_seconds,
    _format_time,
    _ordered_config_names,
    _scan_runtime_cells,
)


def snapshot(name: str) -> Path:
    """Copy every runtime comparison.json into the named baseline tree."""
    dest_root = BASELINES_ROOT / name
    cells = _scan_runtime_cells(RUNTIME_ROOT)
    if not cells:
        sys.stderr.write(f"no comparison.json found under {RUNTIME_ROOT}\n")
        raise SystemExit(1)
    for cell in cells:
        dest = dest_root.joinpath(*cell.cell) / "comparison.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cell.path, dest)
    return dest_root


def render_md(name: str, dest_root: Path) -> Path:
    """One-page markdown: every cell × config headline in the baseline."""
    cells = _scan_runtime_cells(dest_root)
    config_names = _ordered_config_names(cells)

    lines = [
        f"# {name}",
        "",
        f"Named baseline snapshot ({len(cells)} cells; convention: "
        "[`design_lock_in.md`](../../notes/design_lock_in.md)). "
        "Full-pipeline per-call cost per cell × config; `(vmap …)` is the "
        "vmap per-call where measured.",
        "",
        "| Cell | " + " | ".join(config_names) + " |",
        "|" + "|".join(["---"] * (len(config_names) + 1)) + "|",
    ]
    for cell in cells:
        cfgs = cell.configs
        row = [f"`{cell.cell_id}`"]
        for cname in config_names:
            cfg = cfgs.get(cname, {})
            headline = _format_time(_config_headline_seconds(cfg))
            vmap = _config_vmap_seconds(cfg)
            if vmap is not None:
                headline += f" (vmap {_format_time(vmap)})"
            row.append(headline)
        lines.append("| " + " | ".join(row) + " |")

    md_path = dest_root / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Baseline name, e.g. PreOptimizationTimes.")
    args = parser.parse_args(argv)

    dest_root = snapshot(args.name)
    md_path = render_md(args.name, dest_root)
    n = len(_scan_runtime_cells(dest_root))
    print(f"baseline '{args.name}': {n} cell(s) snapshotted -> {dest_root}")
    print(f"rendered {md_path}")
    print("run `python scripts/build_readme.py` to refresh the dashboard's baseline column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
