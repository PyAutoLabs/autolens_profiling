"""
Interferometer Streaming Scaling: Both Paths on One Figure
=========================================================

Reads the ``accumulate`` and ``in_memory`` result JSONs of one release from
``results/streaming_scaling/`` and draws wall time and peak RSS against N_vis
for both paths: the streamed ``from_stream`` rows (one line per chunk size) and
the in-memory ``apply_sparse_operator`` rows (one line per arm, with failures as
crosses at the RSS they reached). It runs no measurement.

Output
------
``results/streaming_scaling/streaming_scaling_local_cpu_v<version>.png``

Usage::

    python scripts/interferometer/delaunay/streaming_plot_scaling.py [--version 2026.10.4.1+2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _streaming  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    parser.add_argument("--results-dir", default=str(_streaming.RESULTS_DIR))
    args = parser.parse_args()
    results = Path(args.results_dir)
    version = args.version or _streaming.release_version()

    accumulate = json.loads((results / f"accumulate_local_cpu_v{version}.json").read_text())
    in_memory = json.loads((results / f"in_memory_local_cpu_v{version}.json").read_text())

    series: dict[str, list[dict]] = {}
    for row in accumulate["rows"]:
        series.setdefault(f"streamed, chunk {row['chunk']}", []).append(row)
    for name, arm in in_memory["arms"].items():
        label = (
            "in-memory, defaults"
            if name == "default"
            else f"in-memory, nufft_chunk_size={arm['nufft_chunk_size']}"
        )
        series[label] = arm["rows"]

    cap = accumulate.get("cli", {}).get("cap_gb", _streaming.CAP_GB_DEFAULT)
    _streaming.plot_series(
        results / f"streaming_scaling_local_cpu_v{version}.png",
        series,
        f"Interferometer streamed vs in-memory, local CPU, {cap} GB RLIMIT_AS, v{version}",
    )


if __name__ == "__main__":
    main()
