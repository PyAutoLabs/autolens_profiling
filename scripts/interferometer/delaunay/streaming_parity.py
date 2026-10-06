"""
Interferometer Streaming Scaling: log_evidence Parity, Streamed vs In-Memory
===========================================================================

Does the array-free dataset give the same answer? The cell compares the
``log_evidence`` of a 20 x 20 rectangular (``RectangularUniform``) sparse
inversion with constant regularization (coefficient 1.0) on the same seeded
visibilities, built two ways:

- ``stream``: ``Interferometer.from_stream`` (chunk ``--chunk``).
- ``memory``: ``Interferometer(...).apply_sparse_operator(...)``. The in-memory
  arm uses ``--nufft-chunk-size`` (65536 by default), because the default
  builder does not fit under the 10 GB cap at 4e6 visibilities (see
  ``in_memory.py``). Pass ``--nufft-chunk-size 0`` for library defaults.

Each arm runs in its own fresh capped child (``_streaming.run_child``). The
correctness result is ``|Δ log_evidence|`` in nats, plus the relative gap. The
two arms' build times and peak RSS are recorded beside it, not mixed into it.

Output
------
``results/streaming_scaling/parity_local_cpu_v<version>.json``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _streaming  # noqa: E402

from _profile_cli import parse_profile_cli, resolve_output_paths  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)


def main() -> None:
    cli = parse_profile_cli()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n-vis", default="4e6")
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--nufft-chunk-size", type=int, default=65536)
    parser.add_argument("--cap-gb", type=float, default=_streaming.CAP_GB_DEFAULT)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = cli.parse_cell_args(parser)

    log_dir = _streaming.ROOT / "output" / "streaming_scaling" / "parity"
    comparisons = []
    for n in _streaming.parse_int_list(args.n_vis):
        stream = _streaming.run_child(
            "stream_evidence",
            n,
            args.chunk,
            cap_gb=args.cap_gb,
            timeout_s=args.timeout,
            log_dir=log_dir,
        )
        memory = _streaming.run_child(
            "memory_evidence",
            n,
            args.chunk,
            cap_gb=args.cap_gb,
            timeout_s=args.timeout,
            nufft_chunk_size=args.nufft_chunk_size or None,
            log_dir=log_dir,
        )
        entry = {"n_vis": n, "stream": stream, "memory": memory}
        if stream.get("outcome") == "OK" and memory.get("outcome") == "OK":
            a, b = stream["log_evidence"], memory["log_evidence"]
            entry["abs_gap_nats"] = abs(a - b)
            entry["rel_gap"] = abs(a - b) / abs(b)
        comparisons.append(entry)

    version = _streaming.release_version()
    json_path, _png = resolve_output_paths(
        cli, _streaming.RESULTS_DIR, f"parity_local_cpu_v{version}", cell="parity"
    )
    payload = {
        "cell": "interferometer/streaming_scaling/parity",
        "question": "log_evidence of a 20x20 rectangular sparse inversion, streamed vs in-memory",
        "release_version": version,
        "inversion": {
            "mesh": f"RectangularUniform{_streaming.MESH_SHAPE}",
            "regularization": f"Constant(coefficient={_streaming.REGULARIZATION_COEFFICIENT})",
            "fit": "aa.m.MockFitInterferometer",
        },
        "comparisons": comparisons,
        **_streaming.run_metadata(vars(args)),
    }
    _streaming.write_json(json_path, payload)


if __name__ == "__main__":
    main()
