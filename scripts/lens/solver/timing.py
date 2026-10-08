"""
Linear-solver timing: batched (jit(vmap)) per-evaluation cost on the solver corpus
===================================================================================

Times the released raw PDIP (``pdip_raw``) against the Jacobi-preconditioned PDIP
(``pdip_jacobi``) as a sampler would call them: ``jax.jit(jax.vmap(solve))`` over a batch of B
systems (``_solvers.batched_kernel``), fp64, at B = 1 / 16 / 50 by default. ``fnnls`` may be
added as host-CPU context; it has no batched form, so its "batch" is a Python loop over the B
lanes (``"looped": true``).

Batches
-------
Multi-system groups are pooled in manifest order (by default ``slam_fixture_571`` +
``slam48_hst``, 56 distinct same-model SLaM systems) and a batch of B takes the first B *distinct*
systems, so lanes diverge in iteration count the way a real Nautilus batch does. A single-system
group (``euclid_vis_lp``) is tiled B times and its rows carry ``"tiled": true`` — its lanes are
identical, so they say nothing about lane divergence. Every row records its lane system names.

Timing
------
The stacked arrays are put on the device once, outside every timed call. ``compile_s`` is the
wall of the first call of each (candidate, batch) config — trace + XLA compile + one execution
(plus any persistent-compile-cache effect, see ``device.cache_fresh``) — and is recorded
separately; it never enters a steady figure. jit compiles once per (candidate, batch shape), so
when two families share a shape only the first one's first call compiles; the other's row has
``"compiled_here": false`` and its ``compile_s`` is just a first execution. Steady cost then comes from ``--rounds`` interleaved rounds: each round calls
every config once, in a fixed order, blocking on the result. Rows report ``wall_min_ms`` /
``wall_median_ms`` of a whole batched call and ``per_eval_min_ms`` / ``per_eval_median_ms`` =
those divided by B.

A vmapped ``while_loop`` runs until its slowest lane stops, so each row records the per-lane
``iterations``, their median and the batch maximum (what the batch pays for), ``n_unconverged``
and the per-lane ``flux_inactive_rel`` against the stored fnnls ``x_ref`` (``_metrics.metrics``).
As a guard that the batched path solves what the unbatched cells solve, every distinct lane
system is also solved unbatched (``candidate.fn``, the ``accuracy.py`` path) and the row records
the per-lane differences (``guard``). A timing is not an admissibility result.

Run from the repo root::

    python scripts/lens/solver/timing.py                         # CPU, defaults
    python scripts/lens/solver/timing.py --device gpu --candidates pdip_raw pdip_jacobi fnnls
    python scripts/lens/solver/timing.py --batch-sizes 1 16 50 --rounds 7 --output-dir /tmp/x

Output
------
``results/lens/solver/timing_summary_<corpus>[_gpu]_v<version>.{json,png}``
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

import argparse  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_GROUPS = ["slam_fixture_571", "slam48_hst", "euclid_vis_lp"]
DEFAULT_CANDIDATES = ["pdip_raw", "pdip_jacobi"]
DEFAULT_BATCH_SIZES = [1, 16, 50]
DEFAULT_ROUNDS = 7


def _parse_args(argv=None):
    """``_driver.parse_cli`` plus this cell's ``--batch-sizes`` / ``--rounds``."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    pre.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    known, rest = pre.parse_known_args(_sys.argv[1:] if argv is None else argv)
    args = _driver.parse_cli(
        __doc__.splitlines()[1] + " (also: --batch-sizes B [B ...], --rounds N)",
        DEFAULT_CANDIDATES,
        argv=rest,
    )
    if args.groups is None:
        args.groups = list(DEFAULT_GROUPS)
    if any(b < 1 for b in known.batch_sizes) or known.rounds < 1:
        raise SystemExit("--batch-sizes and --rounds must be >= 1")
    args.batch_sizes = sorted(set(known.batch_sizes))
    args.rounds = known.rounds
    return args


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def _families(systems):
    """``[(family_name, [System, ...]), ...]``: multi-system groups pooled, singletons alone."""
    by_group: dict[str, list] = {}
    for s in systems:
        by_group.setdefault(s.group, []).append(s)
    pooled = [g for g, ss in by_group.items() if len(ss) > 1]
    out = []
    if pooled:
        out.append(("+".join(pooled), [s for g in pooled for s in by_group[g]]))
    out += [(g, ss) for g, ss in by_group.items() if len(ss) == 1]
    return out


def _lanes(pool, batch_size):
    """The first ``batch_size`` distinct systems of ``pool``, cycled if the pool is smaller."""
    lanes = [pool[i % len(pool)] for i in range(batch_size)]
    return lanes, batch_size > len(pool)


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def _jax_call(cand, lanes, target_kappa):
    """A zero-argument blocking call of the batched kernel on device-resident stacked arrays."""
    import jax
    import jax.numpy as jnp

    Q = jax.device_put(jnp.asarray(np.stack([s.Q for s in lanes]), dtype=jnp.float64))
    q = jax.device_put(jnp.asarray(np.stack([s.q for s in lanes]), dtype=jnp.float64))
    jax.block_until_ready((Q, q))
    f = _solvers.batched_kernel(cand, float(target_kappa))

    def call():
        return jax.block_until_ready(f(Q, q))

    def unpack(out):
        x, conv, it = out
        return (
            np.asarray(x, dtype=np.float64),
            [bool(v) for v in np.asarray(conv)],
            [int(v) for v in np.asarray(it)],
        )

    return call, unpack


def _host_call(cand, lanes, target_kappa):
    """NumPy candidates: a Python loop over the lanes (no batched form exists)."""
    Qs = [np.asarray(s.Q, dtype=np.float64) for s in lanes]
    qs = [np.asarray(s.q, dtype=np.float64) for s in lanes]

    def call():
        return [cand.kernel(Q, q, target_kappa=target_kappa) for Q, q in zip(Qs, qs)]

    def unpack(out):
        x = np.stack([np.asarray(o[0], dtype=np.float64) for o in out])
        its = [o[1].get("outer_iterations") for o in out]
        return x, [True] * len(out), [None if v is None else int(v) for v in its]

    return call, unpack


def _timed(call):
    t0 = time.perf_counter()
    out = call()
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _unbatched(cand, system, target_kappa, cache):
    key = (cand.name, system.group, system.name)
    if key not in cache:
        x, stats = cand.fn(system.Q, system.q, system.meta, target_kappa=target_kappa)
        m = _metrics_of(x, system)
        cache[key] = {
            "flux_inactive_rel": m["flux_inactive_rel"],
            "converged": stats.get("converged"),
            "iterations": stats.get("iterations"),
        }
    return cache[key]


def _metrics_of(x, system):
    import _metrics

    return _metrics.metrics(x, system)


def _lane_record(cand, lanes, x, conv, its, target_kappa, cache):
    import _metrics

    fir, sig, dfir, it_diff, conv_diff = [], [], [], 0, 0
    for i, s in enumerate(lanes):
        m = _metrics_of(x[i], s)
        fir.append(m["flux_inactive_rel"])
        sig.append(m["amp_rel_max_sig"])
        ref = _unbatched(cand, s, target_kappa, cache)
        if m["flux_inactive_rel"] is not None and ref["flux_inactive_rel"] is not None:
            dfir.append(abs(m["flux_inactive_rel"] - ref["flux_inactive_rel"]))
        it_diff += int(its[i] != ref["iterations"])
        conv_diff += int(conv[i] != ref["converged"])
    iters = [v for v in its if v is not None]
    return {
        "iterations": its,
        "median_iterations": _metrics.median_or_none(iters),
        "max_iterations": max(iters, default=None),
        "converged": conv,
        "n_unconverged": sum(1 for c in conv if c is False),
        "flux_inactive_rel": fir,
        "worst_flux_inactive_rel": _metrics.worst_abs(fir),
        "median_flux_inactive_rel": _metrics.median_or_none(fir),
        "worst_amp_rel_max_sig": _metrics.worst_abs(sig),
        "guard": {
            "against": "the same lane solved unbatched by candidate.fn (the accuracy.py path)",
            "max_abs_delta_flux_inactive_rel": max(dfir, default=None),
            "n_lanes_iterations_differ": it_diff,
            "n_lanes_converged_differ": conv_diff,
        },
    }


def _plot(chart_path, rows, title):
    cmap = plt.get_cmap("tab10")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    keys = sorted({(r["candidate"], r["batch_family"]) for r in rows})
    for i, (cand, fam) in enumerate(keys):
        sel = sorted(
            (r for r in rows if r["candidate"] == cand and r["batch_family"] == fam),
            key=lambda r: r["batch_size"],
        )
        bs = [r["batch_size"] for r in sel]
        tiled = any(r["tiled"] for r in sel)
        ls = "--" if tiled else "-"
        lbl = f"{cand} — {fam}{' (tiled)' if tiled else ''}"
        ax1.plot(bs, [r["per_eval_min_ms"] for r in sel], ls, marker="o", color=cmap(i), label=lbl)
        ax1.plot(
            bs, [r["per_eval_median_ms"] for r in sel], ls, marker=".", alpha=0.35, color=cmap(i)
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("batch size B (jit(vmap); fnnls = host loop)")
    ax1.set_ylabel("ms per evaluation (wall / B)")
    ax1.set_title("steady per-eval cost: min (solid marker), median (faint)")
    ax1.legend(fontsize=7)

    labels = [
        f"{r['candidate']}\n{r['batch_family'].split('+')[0]}\nB={r['batch_size']}" for r in rows
    ]
    colors = [cmap(keys.index((r["candidate"], r["batch_family"]))) for r in rows]
    hatch = ["//" if r["compiled_here"] is False else None for r in rows]
    bars = ax2.bar(range(len(rows)), [r["compile_s"] for r in rows], color=colors)
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(labels, fontsize=5, rotation=90)
    ax2.set_yscale("log")
    ax2.set_ylabel("first-call wall (s): trace + compile + one run")
    ax2.set_title("compile cost, separate from steady cost (hatched: shape already compiled)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    unknown = [c for c in args.candidates if c not in _solvers.CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidate(s) {unknown}; known: {list(_solvers.CANDIDATES)}")
    systems = _corpus.load_corpus(args.groups)
    label = _driver.corpus_label(args.groups, _corpus.group_names())
    families = _families(systems)

    # Configs in a fixed order: candidate, family, batch size.
    configs = []
    for name in args.candidates:
        cand = _solvers.CANDIDATES[name]
        for fam, pool in families:
            for b in args.batch_sizes:
                lanes, cycled = _lanes(pool, b)
                make = _jax_call if cand.backend == "jax" else _host_call
                call, unpack = make(cand, lanes, args.target_kappa)
                configs.append(
                    {
                        "cand": cand,
                        "family": fam,
                        "pool_size": len(pool),
                        "b": b,
                        "lanes": lanes,
                        "tiled": len(pool) == 1 or cycled,
                        "call": call,
                        "unpack": unpack,
                        "walls": [],
                    }
                )

    # Compile (first call) per config, recorded on its own. jit caches one executable per
    # (candidate, batch shape), so only the first config of each (candidate, B, n) compiles; a
    # later family of the same shape reuses it and its first call is flagged not-a-compile.
    seen = set()
    for c in configs:
        key = (c["cand"].name, c["b"], c["lanes"][0].n)
        # None for host-loop candidates: nothing is jit-compiled, the first call is just a call.
        c["compiled_here"] = (key not in seen) if c["cand"].backend == "jax" else None
        seen.add(key)
        out, c["compile_s"] = _timed(c["call"])
        c["x"], c["conv"], c["its"] = c["unpack"](out)
        print(
            f"  compile {c['cand'].name:<12s} {c['family']:<28s} B={c['b']:<3d} "
            f"{c['compile_s']:.3f} s  max_it={max((i for i in c['its'] if i is not None), default=None)}",
            flush=True,
        )

    # Steady: interleaved rounds, each config once per round, fixed order.
    for r in range(args.rounds):
        for c in configs:
            _, wall = _timed(c["call"])
            c["walls"].append(wall)
        print(f"  round {r + 1}/{args.rounds} done", flush=True)

    cache: dict = {}
    rows = []
    for c in configs:
        cand, b = c["cand"], c["b"]
        walls_ms = [1.0e3 * w for w in c["walls"]]
        wmin, wmed = min(walls_ms), statistics.median(walls_ms)
        row = {
            "candidate": cand.name,
            "backend": cand.backend,
            "looped": cand.backend != "jax",
            "batch_family": c["family"],
            "batch_size": b,
            "tiled": c["tiled"],
            "pool_size": c["pool_size"],
            "lanes": [f"{s.group}/{s.name}" for s in c["lanes"]],
            "n_distinct_lanes": len({(s.group, s.name) for s in c["lanes"]}),
            "compile_s": c["compile_s"],
            "compiled_here": c["compiled_here"],
            "walls_ms": walls_ms,
            "wall_min_ms": wmin,
            "wall_median_ms": wmed,
            "per_eval_min_ms": wmin / b,
            "per_eval_median_ms": wmed / b,
            **_lane_record(cand, c["lanes"], c["x"], c["conv"], c["its"], args.target_kappa, cache),
        }
        rows.append(row)
        print(
            f"  {cand.name:<12s} {c['family']:<28s} B={b:<3d}{' tiled' if c['tiled'] else '      '} "
            f"per-eval min {row['per_eval_min_ms']:.4f} ms  median {row['per_eval_median_ms']:.4f} ms  "
            f"compile {c['compile_s']:.2f} s  it med/max {row['median_iterations']}/"
            f"{row['max_iterations']}  unconv {row['n_unconverged']}  "
            f"guard dFIR {row['guard']['max_abs_delta_flux_inactive_rel']} "
            f"it-diff {row['guard']['n_lanes_iterations_differ']}",
            flush=True,
        )

    summary = _driver.summary_header("timing", args, systems, label)
    cfg = summary["configuration"]
    cfg.pop("repeats", None)
    cfg["batch_sizes"] = list(args.batch_sizes)
    cfg["rounds"] = args.rounds
    cfg["batched_kernel"] = "jax.jit(jax.vmap(candidate body)) via _solvers.batched_kernel"
    cfg["batch_families"] = {fam: [f"{s.group}/{s.name}" for s in pool] for fam, pool in families}
    cfg["timing_method"] = (
        "inputs device-resident before timing; compile_s = first call per config (separate); "
        "steady = `rounds` interleaved rounds, every config once per round in a fixed order, "
        "blocking; per_eval = wall / batch_size"
    )
    summary["candidates"] = {
        name: {
            "description": _solvers.CANDIDATES[name].description,
            "library_function": _solvers.CANDIDATES[name].library_function,
            "backend": _solvers.CANDIDATES[name].backend,
        }
        for name in args.candidates
    }
    summary["rows"] = rows

    json_path, png_path = _driver.output_paths(
        "timing", label, summary["autolens_version"], args.device, args.output_dir
    )
    _driver.write_json(json_path, summary)
    _plot(
        png_path,
        rows,
        f"linear-solver batched timing — {label} — {args.device} — v{summary['autolens_version']}",
    )
    print(f"  wrote {png_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
