"""
Cluster PointSolver: Library Solver Figure for a Sweep Row (cluster-pointsolver phase 1)
=======================================================================================

Post-processing for ``robust_settings_sweep.py`` (autolens_profiling#436): render
PyAutoLens's own ``point_solver.png`` visualizer
(``autolens.plot.plot_point_solver``, the readable cluster version of
PyAutoLens#795) for one sweep row on one judged draw and source, so a miss can be
seen in the solver's own refinement steps. Local render only, never a timing.

Usage::

    python scripts/cluster/image_plane/solver_figure.py <sweep.json> \\
        --draw 53 73 81 [--config control production_s0.5_p0.001] \\
        [--source K ...] [--output-dir DIR]

``--source`` defaults to every source of the draw with a non-covered root in any
requested row. One PNG per (row, draw, source):
``solver_figure_<dataset>[_pilot]_<config_name>_<row>_draw<N>_source<K>.png``.

What is drawn
-------------

The draw's tracer is rebuilt exactly as ``oracle_figure.py`` does
(``CASE.tracer_from_vector(draws.vectors[N])``); the solver is rebuilt from the
row's stored ``static`` block (limits, scale, precision; ``PointSolver.for_grid``
for the production construction, ``for_limits_and_scale`` otherwise) and the
source-plane coordinate is the oracle's stored ``beta``. The visualizer re-solves
with NumPy (the sweep's ``control_numpy`` row misses the same roots as the JIT
rows, so the NumPy steps are the row's steps). The oracle's accepted roots above
the magnification threshold are passed as the figure's ``positions`` (hollow
circles; the legend's "observed" label means "oracle root" here), so a missed
root is a circle with no solved image on it.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import oracle_figure as of  # noqa: E402

DEFAULT_CONFIGS = ["control", "production_s0.5_p0.001"]


def figure_name(summary: dict, row: str, draw: int, source: int) -> str:
    return f"solver_figure_{of.output_stem(summary).replace('oracle_figure_', '')}_{row}_draw{draw}_source{source}"


def problem_sources(summary: dict, rows, draw: int) -> list:
    """Sources of ``draw`` with a non-covered root in any of ``rows``."""
    out = set()
    for name in rows:
        for entry in summary["rows"][name].get("per_draw") or []:
            if entry["draw"] != draw:
                continue
            for k, s in enumerate(entry["sources"]):
                if any(r["verdict"] != "covered" for r in s.get("roots", [])):
                    out.add(k)
    return sorted(out)


def oracle_positions(summary: dict, draw: int, source: int, threshold: float) -> np.ndarray:
    o = of.oracle_draw(summary, draw)
    roots = [
        r["position"]
        for r in o["sources"][source]["roots"]
        if r["accepted"] and abs(r["magnification"]) >= threshold
    ]
    return np.asarray(roots, dtype=float).reshape(-1, 2)


def make_solver(summary: dict, row: str):
    import autolens as al

    static = summary["rows"][row]["static"]
    cfg = summary["rows"][row]["config"]
    prod = summary["production_solver"]
    common = dict(
        pixel_scale_precision=float(static["pixel_scale_precision"]),
        magnification_threshold=float(prod["magnification_threshold"]),
        neighbor_degree=int(cfg.get("neighbor_degree", 1)),
        use_jax=False,
    )
    if cfg["extent"] == "production" and float(static["scale"]) == float(prod["pixel_scale"]):
        grid = al.Grid2D.uniform(
            shape_native=tuple(prod["shape_native"]), pixel_scales=prod["pixel_scale"]
        )
        return al.PointSolver.for_grid(grid=grid, **common)
    y_min, y_max, x_min, x_max = static["limits"]
    return al.PointSolver.for_limits_and_scale(
        y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max, scale=float(static["scale"]), **common
    )


def render(summary: dict, rows, draws, sources, out_dir: Path) -> list:
    import autolens.plot as aplt

    case = of.rebuild_case(summary)
    threshold = float(summary["production_solver"]["magnification_threshold"])
    written = []
    for draw in draws:
        tracer = of.draw_tracer(case, summary, draw)
        o = of.oracle_draw(summary, draw)
        for k in sources if sources is not None else problem_sources(summary, rows, draw):
            beta = tuple(o["sources"][k]["beta"])
            z = float(summary["sources"][k]["redshift"])
            positions = oracle_positions(summary, draw, k, threshold)
            for row in rows:
                name = figure_name(summary, row, draw, k)
                aplt.plot_point_solver(
                    solver=make_solver(summary, row),
                    tracer=tracer,
                    source_plane_coordinate=beta,
                    positions=positions,
                    plane_redshift=z,
                    output_path=str(out_dir),
                    output_format="png",
                    title_prefix=f"{row} draw {draw} {summary['sources'][k]['name']}:",
                    filename=name,
                )
                written.append(out_dir / f"{name}.png")
                print(f"wrote {written[-1]}", flush=True)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("sweep_json")
    parser.add_argument("--config", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--draw", nargs="+", type=int, required=True)
    parser.add_argument("--source", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default=str(of.DEFAULT_OUTPUT_DIR / "figures"))
    args = parser.parse_args(argv)
    summary = of.load_sweep(args.sweep_json)
    unknown = [c for c in args.config if c not in summary["rows"]]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}")
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    render(summary, args.config, args.draw, args.source, out_dir)


if __name__ == "__main__":
    main()
