"""Aggregate per-config JSONs for a swept likelihood cell into comparison.{json,png}.

Reads every ``<config_name>[_sparse].json`` under a cell's output dir (see
``sweep.py``; default root is ``results/runtime/`` in this repo) and
produces a single ``comparison.json`` whose schema mirrors the historical
``autolens_workspace_developer/jax_profiling/results/jit`` artifacts so the
existing readers (and the OPTIMIZATION_NOTES doc) continue to work.
``_sparse`` rows order after the canonical configs.

The ``comparison.png`` is a log-scale grouped bar chart: one bar per
(step, config), sorted by step cost on the slowest config. The
full-pipeline single-JIT row and the vmap per-call row are appended at
the bottom so the production-cost numbers stand out.

Usage::

    # All cells under the default sweep output root
    python likelihood_runtime/aggregate.py

    # One cell only
    python likelihood_runtime/aggregate.py --cell interferometer/mge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "runtime"


# Stable ordering — keep the same row order as sweep_likelihood + the prior
# imaging precedent so cross-cell tables look uniform.
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
        metavar="CLASS/MODEL",
        help="Only aggregate these cells; default = auto-discover under --output-root.",
    )
    return p.parse_args()


def _discover_cells(output_root: Path) -> list[tuple[str, ...]]:
    """Find every <class>/<model>/[<instrument>/] subdir under output_root.

    Supports two layouts:
    - 2-level (legacy): ``<class>/<model>/<config_name>.json`` — yields ``(class, model)``.
    - 3-level (new):    ``<class>/<model>/<instrument>/<config_name>.json`` — yields ``(class, model, instrument)``.

    The deeper layout lets cells with multiple instruments (interferometer +
    datacube delaunay) maintain a per-instrument comparison view.
    """
    cells: list[tuple[str, ...]] = []
    if not output_root.exists():
        return cells

    def _has_config_json(d: Path) -> bool:
        return any(
            p.stem in _CONFIG_ORDER
            or p.stem.removesuffix("_sparse") in _CONFIG_ORDER
            or p.stem.endswith("_pre_fix")
            for p in d.glob("*.json")
        )

    for cls_dir in sorted(output_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for model_dir in sorted(cls_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if _has_config_json(model_dir):
                cells.append((cls_dir.name, model_dir.name))
            else:
                # Look one level deeper for per-instrument subdirs.
                for inst_dir in sorted(model_dir.iterdir()):
                    if inst_dir.is_dir() and _has_config_json(inst_dir):
                        cells.append((cls_dir.name, model_dir.name, inst_dir.name))
    return cells


def _read_config(json_path: Path) -> dict:
    data = json.loads(json_path.read_text())
    # Normalise field names so downstream rendering is uniform whether the
    # source script wrote ``full_pipeline_single_jit`` (current schema) or
    # the legacy ``full_pipeline_per_call`` (older imaging comparison.json),
    # or ``total_step_by_step_cube`` (datacube — no single-JIT measurement,
    # the cube cost is the sum of per-step JITs).
    if "full_pipeline_per_call" not in data:
        for alt in ("full_pipeline_single_jit", "total_step_by_step_cube"):
            v = data.get(alt)
            if isinstance(v, (int, float)) and np.isfinite(v):
                data["full_pipeline_per_call"] = v
                break
    # Datacube uses steps_cube_cost rather than the per-call ``steps`` dict.
    if "steps" not in data and "steps_cube_cost" in data:
        data["steps"] = data["steps_cube_cost"]
    # Add config_name from filename if absent.
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
    rows = [("config", "backend", "full_pipeline", "vmap_per_call", "log_lik_eager")]
    for name, cfg in comparison["configs"].items():
        backend = cfg.get("device", {}).get("backend", "?")
        full = cfg.get("full_pipeline_per_call") or cfg.get("full_pipeline_single_jit")
        vmap = None
        vmap_info = cfg.get("vmap")
        if isinstance(vmap_info, dict):
            vmap = vmap_info.get("per_call")
        loglik = cfg.get("log_likelihood_eager") or cfg.get("log_likelihood", {}).get("eager_numpy")
        rows.append(
            (
                name,
                str(backend),
                _format_seconds(full),
                _format_seconds(vmap),
                f"{loglik:.4f}" if isinstance(loglik, (int, float)) else "—",
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

    # Collect a stable list of step names across configs.
    step_union: list[str] = []
    for cfg in configs.values():
        for step in cfg.get("steps", {}):
            if step not in step_union:
                step_union.append(step)
    if not step_union:
        # No per-step data — render just the full_pipeline + vmap bars.
        _render_simple_png(configs, cell_id, png_path)
        return

    config_names = list(configs.keys())
    n_steps, n_cfgs = len(step_union), len(config_names)
    # Drop configs that have no positive step data — log scale can't render NaN/0.
    config_names = [
        c
        for c in config_names
        if any(
            isinstance(v, (int, float)) and np.isfinite(v) and v > 0
            for v in configs[c].get("steps", {}).values()
        )
    ]
    n_cfgs = len(config_names)
    if n_cfgs == 0:
        _render_simple_png(configs, cell_id, png_path)
        return
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * n_steps + 1.5)))

    cmap = plt.get_cmap("tab10")
    bar_height = 0.8 / n_cfgs

    y_step = np.arange(n_steps)
    for j, cname in enumerate(config_names):
        cfg = configs[cname]
        steps = cfg.get("steps", {})
        values = [steps.get(s, np.nan) for s in step_union]
        offset = (j - (n_cfgs - 1) / 2) * bar_height
        ax.barh(
            y_step + offset,
            values,
            height=bar_height,
            label=cname,
            color=cmap(j % cmap.N),
            edgecolor="white",
        )

    ax.set_yticks(y_step)
    ax.set_yticklabels(step_union, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Time per call (s, log scale)")
    ax.set_title(f"{cell_id}  — per-step JIT timings", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _render_simple_png(configs: dict, cell_id: str, png_path: Path) -> None:
    """Fallback PNG when no per-step breakdown is available (e.g. interferometer)."""
    config_names = list(configs.keys())
    full_vals = []
    vmap_vals = []
    for cname in config_names:
        cfg = configs[cname]
        full_vals.append(
            cfg.get("full_pipeline_per_call") or cfg.get("full_pipeline_single_jit") or np.nan
        )
        vmap_info = cfg.get("vmap")
        vmap_vals.append(vmap_info.get("per_call") if isinstance(vmap_info, dict) else np.nan)

    full_arr = np.array(full_vals, dtype=float)
    vmap_arr = np.array(vmap_vals, dtype=float)
    has_full = np.any(np.isfinite(full_arr) & (full_arr > 0))
    has_vmap = np.any(np.isfinite(vmap_arr) & (vmap_arr > 0))
    if not (has_full or has_vmap):
        return  # nothing to plot

    n = len(config_names)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * n + 1.5)))
    if has_full:
        ax.barh(y - 0.2, full_arr, height=0.35, label="full_pipeline (single JIT)", color="#4C72B0")
    if has_vmap:
        ax.barh(y + 0.2, vmap_arr, height=0.35, label="vmap per_call", color="#55A868")
    ax.set_yticks(y)
    ax.set_yticklabels(config_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Time per call (s, log scale)")
    ax.set_title(f"{cell_id}  — full pipeline vs vmap", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
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
            if len(parts) not in (2, 3):
                sys.stderr.write(
                    f"bad --cell argument: {spec!r} (expected class/model or class/model/instrument)\n"
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
