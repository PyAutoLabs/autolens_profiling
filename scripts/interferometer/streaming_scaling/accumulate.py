"""
Interferometer Streaming Scaling: Accumulation Time and Memory vs N_vis
=======================================================================

How long does ``Interferometer.from_stream`` take to accumulate an array-free
dataset, and how much memory does it hold while it does, as the visibility count
grows towards the 2e8 of a real ALMA cube (PyAutoLabs Discussion #13)?

Each (N_vis, chunk) row is a fresh child process under an ``RLIMIT_AS`` cap
(``_streaming.run_child``) that feeds seeded synthetic chunks through
``from_stream`` on a 400 x 400 / 0.05" circular mask with ``TransformerNUFFT``.
The row records the accumulation wall time (imports excluded), the child's peak
RSS, seconds per 1e6 visibilities, and the outcome.

Per chunk size, a least-squares fit ``wall = a + b * N_vis`` over the OK rows
gives the fixed cost ``a``, the marginal rate ``b`` (s per 1e6 vis) and the
largest relative residual: the linearity statement the 2e8 extrapolation rests on.

When any OK row's rate exceeds ``--profile-threshold`` (5 s per 1e6 vis by
default), a cProfile of ``sparse_terms_from_chunks`` over ``--profile-n-vis``
visibilities at that row's chunk is taken in one more child and its top-10
functions (by own time and by cumulative time) are stored in the JSON.

Timing, memory and hardware are kept in separate fields; no speed score is formed.

Output
------
``results/streaming_scaling/accumulate_local_cpu_v<version>.{json,png}``

Usage::

    python scripts/interferometer/streaming_scaling/accumulate.py \\
        --n-vis 1e6,4e6,1.6e7 --chunks 4096,65536
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _streaming  # noqa: E402

from _profile_cli import parse_profile_cli, resolve_output_paths  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)


def linear_fit(rows: list[dict]) -> dict | None:
    """``wall = a + b * N`` over the OK rows; ``None`` with fewer than two."""
    import numpy as np

    ok = [r for r in rows if r.get("outcome") == "OK"]
    if len(ok) < 2:
        return None
    n = np.array([r["n_vis"] for r in ok], dtype=float)
    t = np.array([r["wall_s"] for r in ok], dtype=float)
    b, a = np.polyfit(n, t, 1)
    predicted = a + b * n
    return {
        "n_points": len(ok),
        "fixed_s": round(float(a), 2),
        "s_per_1e6_vis": round(float(b) * 1e6, 3),
        "max_rel_residual": round(float(np.max(np.abs(t - predicted) / t)), 4),
        "extrapolated_wall_s_at_2e8": round(float(a + b * 2e8), 1),
        "peak_rss_gb_max": round(max(r["peak_rss_mb"] for r in ok) / 1024, 3),
    }


def main() -> None:
    cli = parse_profile_cli()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n-vis", default="1e6,4e6,1.6e7")
    parser.add_argument("--chunks", default="4096,65536")
    parser.add_argument(
        "--extra",
        default="",
        help="Extra N_vis:chunk rows (e.g. 5e7:65536,1e8:65536) run after the grid.",
    )
    parser.add_argument(
        "--budget-s",
        type=float,
        default=2400.0,
        help=(
            "Wall budget for the --extra rows: each runs only if the elapsed cell time "
            "plus its prediction from the grid's linear fit fits inside it; otherwise it "
            "is recorded as SKIPPED_BUDGET."
        ),
    )
    parser.add_argument("--cap-gb", type=float, default=_streaming.CAP_GB_DEFAULT)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--profile-threshold", type=float, default=5.0)
    parser.add_argument("--profile-n-vis", default="1.6384e5")
    args = cli.parse_cell_args(parser)

    grid = [
        (n, k)
        for k in _streaming.parse_int_list(args.chunks)
        for n in _streaming.parse_int_list(args.n_vis)
    ]
    extra = []
    for item in [x for x in args.extra.split(",") if x.strip()]:
        n, k = item.split(":")
        extra.append((int(float(n)), int(k)))

    log_dir = _streaming.ROOT / "output" / "streaming_scaling" / "accumulate"
    t_start = time.perf_counter()
    rows = [
        _streaming.run_child(
            "stream", n, k, cap_gb=args.cap_gb, timeout_s=args.timeout, log_dir=log_dir
        )
        for n, k in grid
    ]
    for n, k in extra:
        fit = linear_fit([r for r in rows if r["chunk"] == k])
        predicted = fit["fixed_s"] + fit["s_per_1e6_vis"] * n / 1e6 if fit else None
        elapsed = time.perf_counter() - t_start
        if predicted is None or elapsed + predicted > args.budget_s:
            rows.append(
                {
                    "mode": "stream",
                    "n_vis": n,
                    "chunk": k,
                    "outcome": "SKIPPED_BUDGET",
                    "predicted_wall_s": None if predicted is None else round(predicted, 1),
                    "elapsed_cell_s": round(elapsed, 1),
                    "budget_s": args.budget_s,
                }
            )
            print(f"skip {n:.0e}:{k}: predicted {predicted} s, elapsed {elapsed:.0f} s")
            continue
        rows.append(
            _streaming.run_child(
                "stream", n, k, cap_gb=args.cap_gb, timeout_s=args.timeout, log_dir=log_dir
            )
        )

    by_chunk: dict[int, list[dict]] = {}
    for r in rows:
        by_chunk.setdefault(r["chunk"], []).append(r)
    fits = {str(k): linear_fit(v) for k, v in sorted(by_chunk.items())}

    slow = [
        r
        for r in rows
        if r.get("outcome") == "OK" and (r.get("s_per_1e6_vis") or 0) > args.profile_threshold
    ]
    profile = None
    if slow:
        chunk = min(r["chunk"] for r in slow)
        n_prof = int(float(args.profile_n_vis))
        record = _streaming.run_child(
            "profile",
            n_prof,
            chunk,
            cap_gb=args.cap_gb,
            timeout_s=args.timeout,
            log_dir=log_dir,
        )
        profile = {
            "trigger": (
                f"{len(slow)} OK row(s) above {args.profile_threshold} s per 1e6 vis; "
                f"profiled the smallest such chunk ({chunk})"
            ),
            "outcome": record.get("outcome"),
            **(record.get("profile") or {}),
        }

    version = _streaming.release_version()
    json_path, png_path = resolve_output_paths(
        cli,
        _streaming.RESULTS_DIR,
        f"accumulate_local_cpu_v{version}",
        cell="accumulate",
    )
    payload = {
        "cell": "interferometer/streaming_scaling/accumulate",
        "question": "Interferometer.from_stream wall time and peak RSS vs N_vis and chunk size",
        "release_version": version,
        "rows": rows,
        "linear_fit_by_chunk": fits,
        "profile": profile,
        **_streaming.run_metadata(vars(args)),
    }
    _streaming.write_json(json_path, payload)
    _streaming.plot_series(
        png_path,
        {f"from_stream chunk {k}": v for k, v in sorted(by_chunk.items())},
        f"Interferometer.from_stream accumulation (local CPU, cap {args.cap_gb} GB AS), v{version}",
    )


if __name__ == "__main__":
    main()
