"""
JAX Profiling: Delaunay Interferometer Likelihood — Per-Step Breakdown (sparse W~ path)
=======================================================================================

Decomposes the JAX likelihood of an interferometer dataset with a Hilbert-1500
Delaunay source into the steps ``InversionInterferometerSparse`` actually runs per
evaluation, and JIT-profiles each one (autolens_profiling#320, interferometer
likelihood campaign 2/3). The harness is shared with the rectangular sibling
``pixelization.py`` and lives in
``scripts/misc/likelihood_breakdown/interferometer_pixelized.py`` — read its
docstring for the step list, the dense comparison arm, the lever sub-rows and the
JSON schema.

Until 2026-09-26 this cell called ``apply_sparse_operator`` but then timed the
**dense** path (transformed mapping matrix, F from its real/imaginary parts), and
its one-shot dense transform (``1500 x 512 x 512`` complex128 at sma) plus the
compiled executables it never freed put it at 14.6 GB RSS on sma. The sparse arm
now has no transformed-mapping-matrix row, the dense arm lives under
``dense_steps`` with a column-chunked transform, and compiled artefacts are freed
between sections.

Setup (aligned with the imaging campaign)
-----------------------------------------

- ``al.image_mesh.Hilbert(pixels=N)`` on the lensed-source adapt image,
  ``al.mesh.Delaunay``; N = 1500, or ``--source-pixels N`` for the N sweep.
- ``--regularization adapt_split`` (default:
  ``AdaptSplit(inner=0.1, outer=10, signal_scale=0.1)``, as imaging since #232) or
  ``constant_split`` (``ConstantSplit(1.0)``, the bridge to the v2026.5 rows).
- Isothermal + ExternalShear at the simulator truth, no lens light (the
  interferometer datasets carry none).
- ``TransformerNUFFT`` with the instrument's chunk preset; sparse operator built
  with ``method="nufft"``; fp64 (``--use-mixed-precision`` for the mp arm).
- ``--solver {pdip,certified}`` selects ``Settings(positive_only_solver=...)``;
  the interferometer sparse path honours it (a mapper-only JAX inversion).

Output
------

``results/breakdown/interferometer/delaunay_breakdown_{instrument}_v{version}.json``
(+ PNG); config-tagged runs write ``delaunay_<config>.json`` in
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
        cell="delaunay",
        title="Delaunay Interferometer Likelihood",
        fiducial_source_pixels=harness.DELAUNAY_N_FIDUCIAL,
        build_mesh=harness.build_delaunay_mesh,
        pinned={},
    ),
    _cli,
    _cell_args,
    __file__,
    _profiling_root(),
)
