"""Aggregate per-config JSONs for swept search cells into comparison.{json,png}.

Walks the four-level layout written by ``searches/sweep.py``::

    <output_root>/<sampler>/<dataset_class>/<model>/<instrument>/<config_name>.json

For each ``<instrument>`` directory, emits a ``comparison.json`` (per-config
dict) and a ``comparison.png`` (grouped bar chart of the headline metrics
across configs: total_wall_s, viz_wall_s, sampler_wall_s, time_per_eval_ms).

Usage::

    # All cells under the default output root
    python searches/aggregate.py

    # One cell only
    python searches/aggregate.py --cell nautilus/imaging/mge/hst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "searches"


_CONFIG_ORDER = (
    "local_cpu_fp64",
    "local_cpu_mp",
    "local_gpu_fp64",
    "local_gpu_mp",
    "hpc_a100_fp64",
    "hpc_a100_mp",
)


_METRICS_FOR_BAR_CHART: tuple[tuple[str, str], ...] = (
    ("total_wall_s", "Total wall (s)"),
    ("sampler_wall_s", "Sampler wall (s)"),
    ("viz_wall_s", "Viz wall (s)"),
    ("time_per_eval_ms", "Per-eval (ms)"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"Root output dir. Default: {_DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--cell",
        nargs="+",
        default=None,
        metavar="SAMPLER/CLASS/MODEL/INSTRUMENT",
        help=("Only aggregate these cells; default = auto-discover under --output-root."),
    )
    return p.parse_args()


def _discover_cells(output_root: Path) -> list[tuple[str, str, str, str]]:
    """Find every <sampler>/<ds>/<model>/<instrument> dir with config JSONs."""
    cells: list[tuple[str, str, str, str]] = []
    if not output_root.exists():
        return cells

    def _has_config_json(d: Path) -> bool:
        return any(p.stem in _CONFIG_ORDER for p in d.glob("*.json"))

    for sampler_dir in sorted(output_root.iterdir()):
        if not sampler_dir.is_dir():
            continue
        for ds_dir in sorted(sampler_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for model_dir in sorted(ds_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for inst_dir in sorted(model_dir.iterdir()):
                    if inst_dir.is_dir() and _has_config_json(inst_dir):
                        cells.append((sampler_dir.name, ds_dir.name, model_dir.name, inst_dir.name))
    return cells


def _read_config(json_path: Path) -> dict:
    data = json.loads(json_path.read_text())
    data.setdefault("config_name", json_path.stem)
    return data


def _aggregate_cell(cell_dir: Path) -> dict:
    configs: dict[str, dict] = {}
    for json_path in sorted(cell_dir.glob("*.json")):
        if json_path.name == "comparison.json":
            continue
        try:
            configs[json_path.stem] = _read_config(json_path)
        except Exception as exc:
            sys.stderr.write(f"  warn: failed to read {json_path}: {exc}\n")

    ordered: dict[str, dict] = {}
    for name in _CONFIG_ORDER:
        if name in configs:
            ordered[name] = configs.pop(name)
    for name in sorted(configs):
        ordered[name] = configs[name]

    return {"configs": ordered}


def _format_seconds(t: float | None) -> str:
    if t is None or not np.isfinite(t):
        return "—"
    if t >= 1.0:
        return f"{t:.2f}s"
    if t >= 1e-3:
        return f"{t * 1e3:.1f}ms"
    return f"{t * 1e6:.0f}μs"


def _get_perf(cfg: dict, key: str) -> float:
    perf = cfg.get("performance", {})
    val = perf.get(key)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    return float("nan")


def _render_table(comparison: dict, cell_id: str) -> str:
    lines = [f"=== {cell_id} ==="]
    rows = [("config", "backend", "total", "sampler", "viz", "per_eval", "log_evidence")]
    for name, cfg in comparison["configs"].items():
        backend = cfg.get("device", {}).get("backend", "?")
        log_evidence = cfg.get("results", {}).get("log_evidence")
        rows.append(
            (
                name,
                str(backend),
                _format_seconds(_get_perf(cfg, "total_wall_s")),
                _format_seconds(_get_perf(cfg, "sampler_wall_s")),
                _format_seconds(_get_perf(cfg, "viz_wall_s")),
                f"{_get_perf(cfg, 'time_per_eval_ms'):.2f}ms",
                f"{log_evidence:.4f}" if isinstance(log_evidence, (int, float)) else "—",
            )
        )
    col_w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        lines.append("  " + "  ".join(s.ljust(w) for s, w in zip(r, col_w)))
    return "\n".join(lines)


def _render_png(comparison: dict, cell_id: str, png_path: Path) -> None:
    configs = comparison["configs"]
    if not configs:
        return

    config_names = list(configs.keys())
    n_cfgs = len(config_names)
    n_metrics = len(_METRICS_FOR_BAR_CHART)

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.35 * n_metrics + 1.5)))
    cmap = plt.get_cmap("tab10")
    bar_height = 0.8 / n_cfgs

    y_metric = np.arange(n_metrics)
    for j, cname in enumerate(config_names):
        cfg = configs[cname]
        values = [_get_perf(cfg, key) for key, _label in _METRICS_FOR_BAR_CHART]
        offset = (j - (n_cfgs - 1) / 2) * bar_height
        ax.barh(
            y_metric + offset,
            values,
            height=bar_height,
            label=cname,
            color=cmap(j % cmap.N),
            edgecolor="white",
        )

    ax.set_yticks(y_metric)
    ax.set_yticklabels([label for _key, label in _METRICS_FOR_BAR_CHART], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Value (log scale; seconds or ms per the row)")
    ax.set_title(f"{cell_id}  — search profiling comparison", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = _parse_args()

    if args.cell:
        cells: list[tuple[str, ...]] = []
        for spec in args.cell:
            parts = spec.split("/")
            if len(parts) != 4:
                sys.stderr.write(
                    f"bad --cell argument: {spec!r} (expected sampler/class/model/instrument)\n"
                )
                return 2
            cells.append(tuple(parts))
    else:
        cells = _discover_cells(args.output_root)

    if not cells:
        sys.stderr.write(f"no cells found under {args.output_root}\n")
        return 1

    for cell_tuple in cells:
        cell_id = "/".join(cell_tuple)
        cell_dir = args.output_root.joinpath(*cell_tuple)
        if not cell_dir.exists():
            sys.stderr.write(f"  skipping {cell_id}: dir missing\n")
            continue

        comparison = _aggregate_cell(cell_dir)
        if not comparison["configs"]:
            sys.stderr.write(f"  skipping {cell_id}: no per-config JSONs found\n")
            continue

        comparison_path = cell_dir / "comparison.json"
        png_path = cell_dir / "comparison.png"
        comparison_path.write_text(json.dumps(comparison, indent=2, default=str))
        _render_png(comparison, cell_id, png_path)

        print(_render_table(comparison, cell_id))
        print(f"  -> {comparison_path}")
        print(f"  -> {png_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
