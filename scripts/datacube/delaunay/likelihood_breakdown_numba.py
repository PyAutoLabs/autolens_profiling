"""
Numba CPU Profiling: Delaunay Datacube Likelihood -- Per-Step Breakdown (standard size)
=======================================================================================

The datacube breakdown on the NumPy / numba route (likelihood-lifecycle 2f,
autolens_profiling#475): the 2b interferometer numba breakdown
(``scripts/misc/likelihood_breakdown/interferometer_pixelized_numpy.py``, ``--standard-size``)
of the cube's lead channel -- which is the 2b interferometer standard cell, since its emission-line
scale is 1 -- projected to the cube by step (decision D4): the steps the ``shared_preloads`` path
builds once on the lead channel (mapper, mapping matrix, curvature matrix) count once, every other
step ``--n-channels`` times, written as ``total_step_by_step_cube`` (the catalogue's
``cube_component_total``). An eager cube evaluation on the same route then gives the
``parity.json`` witness against the cube reference (decision D3). Threads are pinned to 1 before
numpy is imported. The shared driver is ``datacube_standard.run_breakdown``.

Output: the mirrored cell named by ``--output-path`` (required).
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

import os as _os  # noqa: E402

from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
_thread_env = _pin_thread_env(1)
_thread_env["NUMBA_NUM_THREADS_before"] = _os.environ.get("NUMBA_NUM_THREADS")
_os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse  # noqa: E402

if _os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

from likelihood_breakdown import datacube_standard  # noqa: E402
from likelihood_breakdown import interferometer_pixelized as shared  # noqa: E402
from likelihood_breakdown import interferometer_pixelized_numpy as harness  # noqa: E402

if not _cli.standard_size or not _cli.output_path:
    raise SystemExit(f"{__file__}: runs only the standard-size cell; pass --standard-size and --output-path")

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
harness.add_cell_args(_cell_parser)
_cell_parser.add_argument("--n-channels", type=int, default=datacube_standard.N_CHANNELS)

datacube_standard.run_breakdown(
    _cli,
    _cli.parse_cell_args(_cell_parser),
    shared.CellSpec(
        cell="delaunay",
        title="Delaunay Datacube Likelihood (numba / NumPy, lead channel, projected to the cube)",
        fiducial_source_pixels=shared.DELAUNAY_N_FIDUCIAL,
        build_mesh=shared.build_delaunay_mesh,
        pinned={},
    ),
    numba=True,
    thread_env=_thread_env,
    script_file=__file__,
    profiling_root=_profiling_root(),
)
