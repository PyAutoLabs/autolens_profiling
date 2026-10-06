"""
Interferometer Streaming Scaling: In-Memory Sparse Operator Pushed to Failure
=============================================================================

At what visibility count does the in-memory path, building the whole
``Interferometer`` and then calling ``apply_sparse_operator()``, stop fitting
under a memory cap? This is the crossover the array-free ``from_stream`` dataset
exists for.

Each N_vis row runs in a fresh child process under an ``RLIMIT_AS`` cap
(``_streaming.run_child``, 10 GB by default). The child builds the full
visibility arrays from the same seeded chunks that ``accumulate.py`` streams,
then builds the sparse operator. The grid ascends and each arm stops at its
first failure: that N_vis, its error and the RSS it reached are the result.

Two arms:

- ``default``: ``apply_sparse_operator()`` with library defaults. This is the
  call a user makes. The NUFFT precision-operator builder runs over every
  visibility in one pass.
- ``nufft_chunked``: ``apply_sparse_operator(nufft_chunk_size=K)``. This is the
  builder's own memory ceiling (``nufft_precision_operator_from(chunk_size=)``).
  It is still in memory (the full arrays are held), and it shows how much of
  the default arm's failure is the builder's gather buffer rather than the
  visibilities themselves. Pass ``--nufft-chunk-size 0`` to skip it.

Output
------
``results/streaming_scaling/in_memory_local_cpu_v<version>.{json,png}``
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


def run_arm(n_list, chunk, nufft_chunk_size, args, log_dir) -> dict:
    rows = []
    for n in n_list:
        record = _streaming.run_child(
            "memory",
            n,
            chunk,
            cap_gb=args.cap_gb,
            timeout_s=args.timeout,
            nufft_chunk_size=nufft_chunk_size,
            log_dir=log_dir,
        )
        rows.append(record)
        if record.get("outcome") != "OK":
            break
    ok = [r for r in rows if r.get("outcome") == "OK"]
    failed = [r for r in rows if r.get("outcome") != "OK"]
    return {
        "nufft_chunk_size": nufft_chunk_size,
        "rows": rows,
        "largest_ok_n_vis": max((r["n_vis"] for r in ok), default=None),
        "first_failure": (
            {
                "n_vis": failed[0]["n_vis"],
                "outcome": failed[0]["outcome"],
                "error": failed[0].get("error"),
                "peak_rss_gb": failed[0].get("peak_rss_gb"),
            }
            if failed
            else None
        ),
        "untried_n_vis": [n for n in n_list if n not in {r["n_vis"] for r in rows}],
    }


def main() -> None:
    cli = parse_profile_cli()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n-vis", default="1e5,2.5e5,5e5,1e6,4e6,1.6e7,5e7,1e8")
    parser.add_argument(
        "--chunk",
        type=int,
        default=65536,
        help="Generation chunk of the seeded visibilities (matches accumulate.py).",
    )
    parser.add_argument("--nufft-chunk-size", type=int, default=65536)
    parser.add_argument("--cap-gb", type=float, default=_streaming.CAP_GB_DEFAULT)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = cli.parse_cell_args(parser)

    n_list = _streaming.parse_int_list(args.n_vis)
    log_dir = _streaming.ROOT / "output" / "streaming_scaling" / "in_memory"
    arms = {"default": run_arm(n_list, args.chunk, None, args, log_dir)}
    if args.nufft_chunk_size:
        arms["nufft_chunked"] = run_arm(n_list, args.chunk, args.nufft_chunk_size, args, log_dir)

    version = _streaming.release_version()
    json_path, png_path = resolve_output_paths(
        cli, _streaming.RESULTS_DIR, f"in_memory_local_cpu_v{version}", cell="in_memory"
    )
    payload = {
        "cell": "interferometer/streaming_scaling/in_memory",
        "question": (
            "Interferometer(...).apply_sparse_operator() peak RSS and wall vs N_vis; "
            "first N_vis that fails under the cap"
        ),
        "release_version": version,
        "arms": arms,
        **_streaming.run_metadata(vars(args)),
    }
    _streaming.write_json(json_path, payload)
    _streaming.plot_series(
        png_path,
        {
            (
                "apply_sparse_operator()"
                if k == "default"
                else f"nufft_chunk_size={v['nufft_chunk_size']}"
            ): v["rows"]
            for k, v in arms.items()
        },
        f"In-memory apply_sparse_operator (local CPU, cap {args.cap_gb} GB AS), v{version}",
    )


if __name__ == "__main__":
    main()
