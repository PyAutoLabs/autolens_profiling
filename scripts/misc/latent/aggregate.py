"""Aggregate per-config JSONs for a swept latent cell into comparison.{json,png}.

Reads every ``<config_name>.json`` under a cell's output dir (see
``sweep.py``) and produces a single ``comparison.json`` and a
``comparison.png`` bar chart.

The ``comparison.png`` plots first-call vs steady-state vs vmap-per-call time
per config on a log scale. For the ``effective_einstein_radius`` cell, the
chart additionally surfaces the ``closure_cache_first_call_s`` and
``closure_cache_second_call_s`` fields when present, so the LensCalc closure
warm-up cost is immediately visible.

Usage::

    # All latent cells under the default sweep output root
    python latent/aggregate.py

    # One cell only
    python latent/aggregate.py --cell imaging/effective_einstein_radius
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)


import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = _profiling_root()
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "latent"

# Canonical ordering of cells — drives auto-discovery sort order.
_CELLS: list[tuple[str, str]] = [
    ("imaging", "total_lens_flux_mujy"),
    ("imaging", "total_lensed_source_flux_mujy"),
    ("imaging", "total_source_flux_mujy"),
    ("imaging", "magnification"),
    ("imaging", "effective_einstein_radius"),
]

# Stable config ordering across all comparison tables.
_CONFIG_ORDER = (
    "local_cpu_fp64",
    "local_cpu_mp",
    "local_gpu_fp64",
    "local_gpu_mp",
    "hpc_a100_fp64",
    "hpc_a100_mp",
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
        metavar="CLASS/LATENT",
        help="Only aggregate these cells; default = auto-discover under --output-root.",
    )
    return p.parse_args()


def _discover_cells(output_root: Path) -> list[tuple[str, str]]:
    """Find every <class>/<latent> subdir that contains at least one config JSON."""
    if not output_root.exists():
        return []

    def _has_config_json(d: Path) -> bool:
        return any(p.stem in _CONFIG_ORDER for p in d.glob("*.json"))

    found: list[tuple[str, str]] = []
    for cls_dir in sorted(output_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for latent_dir in sorted(cls_dir.iterdir()):
            if latent_dir.is_dir() and _has_config_json(latent_dir):
                found.append((cls_dir.name, latent_dir.name))
    return found


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

    # Reorder by the canonical config order, then any extras at the end.
    ordered: dict[str, dict] = {}
    for name in _CONFIG_ORDER:
        if name in configs:
            ordered[name] = configs.pop(name)
    for name, data in sorted(configs.items()):
        ordered[name] = data

    return {"configs": ordered}


def _format_seconds(t: float | None) -> str:
    if t is None or not np.isfinite(t):
        return "—"
    if t >= 1.0:
        return f"{t:.2f}s"
    if t >= 1e-3:
        return f"{t * 1e3:.1f}ms"
    return f"{t * 1e6:.0f}μs"


def _render_table(comparison: dict, cell_id: str) -> str:
    lines = [f"=== {cell_id} ==="]
    is_einstein = "effective_einstein_radius" in cell_id
    if is_einstein:
        header = (
            "config",
            "eager_time",
            "jit_first",
            "jit_steady",
            "vmap_per_call",
            "cache_first",
            "cache_second",
        )
    else:
        header = ("config", "eager_time", "jit_first", "jit_steady", "vmap_per_call")
    rows = [header]
    for name, cfg in comparison["configs"].items():
        eager_t = cfg.get("eager_time_s")
        jit_first = cfg.get("jit_first_call_s")
        jit_steady = cfg.get("jit_steady_state_s")
        vmap = cfg.get("vmap_per_call_s")
        if is_einstein:
            cc_first = cfg.get("closure_cache_first_call_s")
            cc_second = cfg.get("closure_cache_second_call_s")
            rows.append(
                (
                    name,
                    _format_seconds(eager_t),
                    _format_seconds(jit_first),
                    _format_seconds(jit_steady),
                    _format_seconds(vmap),
                    _format_seconds(cc_first),
                    _format_seconds(cc_second),
                )
            )
        else:
            rows.append(
                (
                    name,
                    _format_seconds(eager_t),
                    _format_seconds(jit_first),
                    _format_seconds(jit_steady),
                    _format_seconds(vmap),
                )
            )
    col_w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        lines.append("  " + "  ".join(s.ljust(w) for s, w in zip(r, col_w)))
    return "\n".join(lines)


def _render_png(comparison: dict, cell_id: str, png_path: Path) -> None:
    """Bar chart: first-call vs steady-state vs vmap-per-call, per config.

    For effective_einstein_radius, also plots closure_cache first/second call
    as additional series when present.
    """
    configs = comparison["configs"]
    if not configs:
        return

    config_names = list(configs.keys())
    is_einstein = "effective_einstein_radius" in cell_id

    # Collect timing series.
    series: dict[str, list[float]] = {
        "jit_first_call": [],
        "jit_steady_state": [],
        "vmap_per_call": [],
    }
    if is_einstein:
        series["closure_cache_first"] = []
        series["closure_cache_second"] = []

    for cname in config_names:
        cfg = configs[cname]
        series["jit_first_call"].append(cfg.get("jit_first_call_s", np.nan))
        series["jit_steady_state"].append(cfg.get("jit_steady_state_s", np.nan))
        series["vmap_per_call"].append(cfg.get("vmap_per_call_s", np.nan))
        if is_einstein:
            series["closure_cache_first"].append(cfg.get("closure_cache_first_call_s", np.nan))
            series["closure_cache_second"].append(cfg.get("closure_cache_second_call_s", np.nan))

    # Drop series that are entirely nan/zero — nothing to plot.
    series = {k: v for k, v in series.items() if any(np.isfinite(x) and x > 0 for x in v)}
    if not series:
        return

    n_cfgs = len(config_names)
    n_series = len(series)
    y = np.arange(n_cfgs)
    bar_height = 0.8 / n_series

    cmap = plt.get_cmap("tab10")
    label_map = {
        "jit_first_call": "JIT first call",
        "jit_steady_state": "JIT steady-state",
        "vmap_per_call": "vmap per-call",
        "closure_cache_first": "closure cache — first",
        "closure_cache_second": "closure cache — second",
    }

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * n_cfgs + 1.5)))
    for j, (key, vals) in enumerate(series.items()):
        arr = np.array(vals, dtype=float)
        offset = (j - (n_series - 1) / 2) * bar_height
        ax.barh(
            y + offset,
            arr,
            height=bar_height,
            label=label_map.get(key, key),
            color=cmap(j % cmap.N),
            edgecolor="white",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(config_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Time per call (s, log scale)")
    ax.set_title(f"{cell_id}  — latent timings", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = _parse_args()

    if args.cell:
        cells: list[tuple[str, str]] = []
        for spec in args.cell:
            parts = spec.split("/")
            if len(parts) != 2:
                sys.stderr.write(f"bad --cell argument: {spec!r} (expected class/latent)\n")
                return 2
            cells.append((parts[0], parts[1]))
    else:
        cells = _discover_cells(args.output_root)

    if not cells:
        sys.stderr.write(f"no cells found under {args.output_root}\n")
        return 1

    for cls, latent in cells:
        cell_id = f"{cls}/{latent}"
        cell_dir = args.output_root / cls / latent
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
