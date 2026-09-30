"""
Linear-solver accuracy: every candidate on every corpus system
===============================================================

Re-solves each captured positive-only system ``(Q, q)`` of the corpus
(``results/lens/solver/corpus/``) with every registered candidate in ``_solvers.py`` and scores
the reconstruction against the stored fnnls reference ``x_ref`` (``_metrics.py``): amplitude
errors, summed-amplitude (flux-proxy) errors on all and on source columns, the relative
objective gap and the scale-normalised KKT residual — next to the solver's own ``converged`` /
``iterations`` and its median warm wall-clock.

The question it answers across releases: *does the positive-only solver the library ships
return the right amplitudes on the systems real models produce, and at what iteration cost?*
A solver can report ``converged`` and still be wrong (a loose tolerance), or be right while
reporting failure (an unreachable tolerance); the reference-anchored metrics separate the two.

Per-(system, candidate) rows plus per-candidate aggregates (worst ``amp_rel_max``, worst
``|flux_rel_source|``, unconverged count, median iterations, median wall). The PNG plots
``amp_rel_max`` against ``kkt_residual_scaled`` per candidate and the iteration histogram.

Run from the repo root::

    python scripts/lens/solver/accuracy.py                      # all groups, default candidates
    python scripts/lens/solver/accuracy.py --groups slam_fixture_571 --candidates pdip_raw fnnls

Output
------
``results/lens/solver/accuracy_summary_<corpus>[_gpu]_v<version>.{json,png}``
"""

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

_sys.path.insert(0, str(_profiling_root()))

import _driver  # noqa: E402

# --device must pin the JAX platform before anything imports jax.
_DEVICE = _driver.select_device_from_argv()

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os  # noqa: E402
import sys as _smoke_sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import _corpus  # noqa: E402
import _solvers  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)


def _fmt(v) -> str:
    return "None" if v is None else f"{v:.3e}"


def _plot(chart_path, rows, aggregates, title):
    candidates = list(aggregates)
    cmap = plt.get_cmap("tab20")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, name in enumerate(candidates):
        pts = [
            (_driver.log_floor(r["kkt_residual_scaled"]), _driver.log_floor(r["amp_rel_max"]))
            for r in rows
            if r["candidate"] == name and r["finite"]
        ]
        if pts:
            xs, ys = zip(*pts)
            ax1.scatter(xs, ys, s=22, color=cmap(i % 20), label=name, alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("kkt_residual_scaled")
    ax1.set_ylabel("amp_rel_max vs fnnls")
    ax1.set_title(f"accuracy vs KKT residual (zeros drawn at {_driver.PLOT_FLOOR:g})")
    ax1.legend(fontsize=7, ncol=2)

    iters = [
        [r["iterations"] for r in rows if r["candidate"] == name and r["iterations"] is not None]
        for name in candidates
    ]
    top = max((max(v) for v in iters if v), default=1)
    bins = range(0, int(top) + 2)
    for i, (name, vals) in enumerate(zip(candidates, iters)):
        if vals:
            ax2.hist(vals, bins=bins, histtype="step", lw=1.5, color=cmap(i % 20), label=name)
    ax2.set_xlabel("iterations (PDIP iters / active-set passes / fnnls outer iters)")
    ax2.set_ylabel("systems")
    ax2.set_title("iteration histogram")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = _driver.parse_cli(__doc__.splitlines()[1], _solvers.ACCURACY_DEFAULT)
    unknown = [c for c in args.candidates if c not in _solvers.CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidate(s) {unknown}; known: {list(_solvers.CANDIDATES)}")
    systems = _corpus.load_corpus(args.groups)
    label = _driver.corpus_label(args.groups, _corpus.group_names())

    rows = []
    for name in args.candidates:
        cand = _solvers.CANDIDATES[name]
        for system in systems:
            row = _driver.evaluate(
                cand, system, repeats=args.repeats, target_kappa=args.target_kappa
            )
            rows.append(row)
            print(
                f"  {name:<22s} {row['system']:<26s} conv={row['converged']!s:<5s} "
                f"it={row['iterations']!s:>4s} amp_rel_max={_fmt(row['amp_rel_max'])} "
                f"wall={row['wall_ms']:.3f} ms",
                flush=True,
            )
    aggregates = {
        name: _driver.aggregate([r for r in rows if r["candidate"] == name])
        for name in args.candidates
    }

    summary = _driver.summary_header("accuracy", args, systems, label)
    summary["candidates"] = {
        name: {
            "description": _solvers.CANDIDATES[name].description,
            "library_function": _solvers.CANDIDATES[name].library_function,
            "backend": _solvers.CANDIDATES[name].backend,
        }
        for name in args.candidates
    }
    summary["aggregates"] = aggregates
    summary["rows"] = rows

    json_path, png_path = _driver.output_paths(
        "accuracy", label, summary["autolens_version"], args.device, args.output_dir
    )
    _driver.write_json(json_path, summary)
    _plot(
        png_path,
        rows,
        aggregates,
        f"linear-solver accuracy — {label} — v{summary['autolens_version']}",
    )
    print(f"  wrote {png_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
