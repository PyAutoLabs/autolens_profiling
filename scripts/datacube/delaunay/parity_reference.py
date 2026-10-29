"""
Parity reference: the cube reference log evidence of the datacube standard cell
===============================================================================

Likelihood-lifecycle 2f, decision D3 (autolens_profiling#475). Every datacube cell writes a
``parity.json`` comparing its cube log likelihood with the cube reference, which follows from the
2b reference of the same (delaunay, instrument, instance): the visibility-chunked eager NumPy dense
reference (2b ruling R6), cached by the 2b reference task at

    output/p2b_references/interferometer_delaunay_<instrument>.json

This script computes that reference (or reuses a valid cache, exactly as
``scripts/interferometer/pixelized/parity_reference.py --likelihood delaunay`` does), then, with
``--results-root``, rewrites the ``parity.json`` of every datacube row of the instrument under
``<results-root>/results/datacube/delaunay/<instrument>/`` that carries a cube
``parity_candidate``. A cell that finishes after the reference reads the cache itself.

Usage::

    python scripts/datacube/delaunay/parity_reference.py --instrument alma \\
        --results-root output/p2f_batch1
    python scripts/datacube/delaunay/parity_reference.py --instrument alma_high \\
        --results-root output/p2f_batch1 --finalise-only
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

import argparse  # noqa: E402
import json  # noqa: E402
import os as _smoke_os  # noqa: E402

from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
instrument = _cli.instrument or "sma"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument(
        "--results-root",
        default=None,
        help="Scratch tree whose datacube rows of this instrument get their parity.json finalised.",
    )
    parser.add_argument("--vis-chunk", type=int, default=None)
    parser.add_argument("--column-chunk", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="recompute a valid cached reference")
    parser.add_argument(
        "--finalise-only",
        action="store_true",
        help="never compute: finalise the rows from an existing cache",
    )
    return parser


_args, _unknown = _parser().parse_known_args()

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

from likelihood_breakdown.datacube_standard import finalise_parity  # noqa: E402
from likelihood_breakdown.interferometer_standard import reference_task  # noqa: E402

_root = _profiling_root()
_doc = reference_task(
    "delaunay",
    instrument,
    profiling_root=_root,
    vis_chunk=_args.vis_chunk,
    column_chunk=_args.column_chunk,
    force=_args.force,
    compute=not _args.finalise_only,
)
if _args.results_root:
    _summary = finalise_parity(instrument, results_root=_Path(_args.results_root), reference=_doc)
    print(json.dumps(_summary, indent=2))
    if _summary["failed"]:
        _sys.exit(4)
