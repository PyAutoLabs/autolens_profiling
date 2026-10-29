"""
Parity reference: the visibility-chunked eager NumPy dense log evidence of an interferometer cell
=================================================================================================

Likelihood-lifecycle 2b, rulings R6 / R7 (autolens_profiling#469). Every interferometer pixelized
cell (``rectangular`` / ``delaunay``, every setup) writes a ``parity.json`` comparing its log
evidence at the standard-size cell's iid draw 0 with this reference: the eager NumPy dense route
(``InversionInterferometerMapping``), independent of the W~ operator, of numba and of JAX, with
the transformed mapping matrix formed per visibility chunk
(``_production_config.interferometer_dense_reference``). At sma a cell computes it in-process; at
alma, alma_high and jvla the library's one-shot dense transform does not fit in memory, so this
script runs it once per (variant, instrument) as its own RAL task and caches it::

    output/p2b_references/interferometer_<variant>_<instrument>.json

keyed on the dataset files' md5, the library revisions and the instance's unit vector, so a cell
only uses a reference that describes it. ``--results-root`` then finalises: every cell row of
that likelihood under ``<results-root>/results/interferometer/<variant>/<instrument>/`` that
carries a ``parity_candidate`` (written by the cells) gets its ``parity.json`` rewritten against
the reference. A cell that finishes after the reference reads the cache itself.

Usage::

    python scripts/interferometer/pixelized/parity_reference.py --instrument jvla \\
        --likelihood delaunay --results-root output/p2b_batch1
    python scripts/interferometer/pixelized/parity_reference.py --instrument alma \\
        --likelihood rectangular --results-root output/p2b_batch1 --finalise-only

A valid cached reference is reused unless ``--force``.
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
    parser.add_argument("--likelihood", choices=("rectangular", "delaunay"), required=True)
    parser.add_argument(
        "--results-root",
        default=None,
        help="Scratch tree whose cell rows of this likelihood get their parity.json finalised.",
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

from likelihood_breakdown.interferometer_standard import (  # noqa: E402
    finalise_parity,
    reference_task,
)

_root = _profiling_root()
_doc = reference_task(
    _args.likelihood,
    instrument,
    profiling_root=_root,
    vis_chunk=_args.vis_chunk,
    column_chunk=_args.column_chunk,
    force=_args.force,
    compute=not _args.finalise_only,
)
if _args.results_root:
    _summary = finalise_parity(
        _args.likelihood, instrument, results_root=_Path(_args.results_root), reference=_doc
    )
    print(json.dumps(_summary, indent=2))
    if _summary["failed"]:
        _sys.exit(4)
