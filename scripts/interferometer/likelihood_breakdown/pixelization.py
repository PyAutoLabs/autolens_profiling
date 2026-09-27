"""
JAX Profiling: Rectangular Interferometer Likelihood — Per-Step Breakdown (sparse W~ path)
==========================================================================================

The rectangular (39 x 39 = 1521 source pixels) sibling of ``delaunay.py``: the
same harness (``scripts/misc/likelihood_breakdown/interferometer_pixelized.py``,
whose docstring documents the steps, the dense comparison arm, the lever
sub-rows and the JSON schema), the same dataset, lens and sparse operator, with
the imaging campaign's rectangular setup (autolens_profiling#320). Only numba
variants of this cell existed before; this is the first JAX one.

Setup
-----

- ``rect_mesh_classes(cli)[1]`` — ``RectangularBilinearAdaptImage`` (or the RTU
  family with ``--rect-mesh rtu``), ``shape=(39, 39)``, ``weight_power=1.0``,
  ``weight_floor=0.0``, weighted by the lensed-source adapt image;
  ``--source-pixels N`` builds the nearest square, ``round(sqrt(N))`` per side.
- ``al.reg.Constant(coefficient=1.0)`` (the rectangular cells' fixed scheme; no
  ``--regularization`` flag).
- Isothermal + ExternalShear at the simulator truth, no lens light;
  ``TransformerNUFFT`` with the instrument's chunk preset; sparse operator with
  ``method="nufft"``; fp64 unless ``--use-mixed-precision``.

Output
------

``results/breakdown/interferometer/pixelization_breakdown_{instrument}_v{version}.json``
(+ PNG); config-tagged runs write ``pixelization_<config>.json`` in
``interferometer/`` for alma and ``interferometer/<instrument>/`` otherwise.
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
import os as _smoke_os  # noqa: E402

from likelihood_breakdown import interferometer_pixelized as harness  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
harness.add_cell_args(_cell_parser)
_cell_args = _cli.parse_cell_args(_cell_parser)

harness.run(
    harness.CellSpec(
        cell="pixelization",
        title="Rectangular Interferometer Likelihood",
        fiducial_source_pixels=harness.RECT_SIDE_FIDUCIAL**2,
        build_mesh=harness.build_rectangular_mesh,
        pinned={},
    ),
    _cli,
    _cell_args,
    __file__,
    _profiling_root(),
)
