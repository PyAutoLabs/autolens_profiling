"""
Linear-solver early stopping: released raw PDIP error vs iteration cap
=======================================================================

Runs the released raw-mode PDIP solve (``pdip_raw``: forward PDIP on the un-preconditioned
system at the data-scaled tolerance) on every corpus system at each iteration cap in
``_solvers.RAW_CAPS`` (``pdip_raw_cap_<N>`` candidates) and scores each result against the
stored fnnls reference. A cap below the natural convergence count returns an unconverged
iterate; this cell measures how wrong that iterate is — the evidence for choosing
``Settings.nnls_max_iter`` (which also caps the worst-case cost of a ``vmap``-ed batch, whose
while-loop runs until the slowest lane stops).

Per-(system, cap) rows plus per-cap aggregates (converged count, worst / median
``amp_rel_max``, worst KKT residual, worst ``|flux_rel_source|``, median wall). The PNG draws
error-vs-cap curves, one line per system.

Run from the repo root::

    python scripts/lens/solver/early_stopping.py
    python scripts/lens/solver/early_stopping.py --candidates pdip_raw_cap_8 pdip_raw_cap_16

Output
------
``results/lens/solver/early_stopping_summary_<corpus>[_gpu]_v<version>.{json,png}``
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


def _cap_of(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _plot(chart_path, rows, caps, title):
    systems = sorted({r["system"] for r in rows})
    panels = (
        ("amp_rel_max", "amp_rel_max vs fnnls"),
        ("kkt_residual_scaled", "kkt_residual_scaled"),
        ("flux_rel_source", "|flux_rel_source|"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.get_cmap("tab10")
    for ax, (key, ylabel) in zip(axes, panels):
        for i, system in enumerate(systems):
            pts = [
                (r["cap"], _driver.log_floor(r[key]))
                for r in rows
                if r["system"] == system and r[key] is not None
            ]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=cmap(i % 10), label=system)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(caps)
        ax.set_xticklabels([str(c) for c in caps])
        ax.set_xlabel("PDIP iteration cap (max_iter)")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = _driver.parse_cli(__doc__.splitlines()[1], _solvers.CAP_CANDIDATES)
    bad = [c for c in args.candidates if c not in _solvers.CAP_CANDIDATES]
    if bad:
        raise SystemExit(f"early_stopping takes cap candidates only; got {bad}")
    names = sorted(args.candidates, key=_cap_of)
    args.candidates = names
    systems = _corpus.load_corpus(args.groups)
    label = _driver.corpus_label(args.groups, _corpus.group_names())

    rows = []
    for name in names:
        cand = _solvers.CANDIDATES[name]
        for system in systems:
            row = _driver.evaluate(
                cand, system, repeats=args.repeats, target_kappa=args.target_kappa
            )
            row["cap"] = _cap_of(name)
            rows.append(row)
        print(f"  cap {_cap_of(name):>4d}: done", flush=True)
    caps = [_cap_of(n) for n in names]
    aggregates = {str(cap): _driver.aggregate([r for r in rows if r["cap"] == cap]) for cap in caps}

    summary = _driver.summary_header("early_stopping", args, systems, label)
    summary["solver"] = {
        "base_candidate": "pdip_raw",
        "caps": caps,
        "description": _solvers.CANDIDATES["pdip_raw"].description,
        "library_function": _solvers.CANDIDATES[names[0]].library_function,
    }
    summary["aggregates"] = aggregates
    summary["rows"] = rows

    json_path, png_path = _driver.output_paths(
        "early_stopping", label, summary["autolens_version"], args.device, args.output_dir
    )
    _driver.write_json(json_path, summary)
    _plot(
        png_path,
        rows,
        caps,
        f"pdip_raw error vs iteration cap — {label} — v{summary['autolens_version']}",
    )
    print(f"  wrote {png_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
