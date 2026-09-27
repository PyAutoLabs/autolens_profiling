"""
Numba CPU Profiling: Rectangular Interferometer Likelihood — Per-Step Breakdown (library dispatch)
==================================================================================================

Decomposes the NumPy likelihood of an interferometer dataset with a 39x39 adaptive rectangular source into the
steps the **library's own dispatch** runs per evaluation, on the numba ``direct_conv``
route (``InversionInterferometerSparseNumba``) and the NumPy FFT route
(``InversionInterferometerSparse(xp=np)``) on identical inputs (autolens_profiling#326,
interferometer likelihood campaign 3/3, phase 1). The harness is shared with the
Delaunay sibling ``delaunay_numba.py`` and lives in
``scripts/misc/likelihood_breakdown/interferometer_pixelized_numpy.py`` — read its docstring
for the arms, the step list, the protocol and the JSON schema. The dataset, mesh, adapt
image and model are the JAX cell ``pixelization.py``'s (built by the same helpers), so the
JAX-CPU arm is that cell run with ``JAX_PLATFORMS=cpu``; this cell records a pointer to its
row.

Until 2026-09-27 this cell timed the recovered numba prototype pack
(``scripts/misc/numba_interferometer/``, ``--kernel``) with a ``Rectangular`` + ``Constant`` setup; the pack and
its bake-off still hold those kernels, but this cell now measures what
``FitInterferometer(..., xp=np)`` actually runs.

Setup
-----

- ``rect_mesh_classes(cli)[1]`` (``RectangularBilinearAdaptImage``, or the RTU family
  with ``--rect-mesh rtu``), ``shape=(39, 39)``, edge-zeroed; ``--source-pixels N`` takes
  the nearest square. ``Constant(1.0)`` regularization; no lens light.
- Threads pinned to 1 (BLAS family + ``NUMBA_NUM_THREADS``) here, before numpy is
  imported; NNLS memo off; iid instance stream (``--n-instances``, default 20).
- ``--numba-gate`` (default 1e9, i.e. forced to admit) is the numba arm's
  ``Settings.interferometer_numba_nnz_per_source_max``; the FFT arm runs at gate 0.
- ``--mask-radius`` overrides the preset's 3.5"; ``--arms`` selects ``numba`` /
  ``numpy_fft``.

Output
------

``results/breakdown/interferometer/pixelization_numba_breakdown_{instrument}_v{version}.json``
(+ PNG); config-tagged runs write ``pixelization_numba_<config>.json`` in ``interferometer/`` for
alma and ``interferometer/<instrument>/`` otherwise; ``_r<radius>`` for a non-default mask.
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

# Threads are pinned BEFORE numpy (and numba) is imported: OpenBLAS / MKL read their
# thread-count variables once, when the shared library loads, and numba reads
# NUMBA_NUM_THREADS at import. Both modules below are stdlib-only at import time.
import os as _os  # noqa: E402

from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
_thread_env = _pin_thread_env(1)
_thread_env["NUMBA_NUM_THREADS_before"] = _os.environ.get("NUMBA_NUM_THREADS")
_os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse  # noqa: E402

from likelihood_breakdown import interferometer_pixelized as shared  # noqa: E402
from likelihood_breakdown import interferometer_pixelized_numpy as harness  # noqa: E402

if _os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
harness.add_cell_args(_cell_parser)
_cell_args = _cli.parse_cell_args(_cell_parser)

harness.run(
    shared.CellSpec(
        cell="pixelization",
        title="Rectangular Interferometer Likelihood (numba / NumPy)",
        fiducial_source_pixels=shared.RECT_SIDE_FIDUCIAL**2,
        build_mesh=shared.build_rectangular_mesh,
        # Prior-median log evidence, numba arm (the FFT arm is checked against the same
        # value): pinned 2026-09-27 from the RAL CPU row (library mains of that date,
        # fiducial mesh, preset mask radius, the May-18 sma lensed_source.fits adapt
        # image shared with the #324 A100 rows). rtol 1e-6 covers the ~1e-8 Delaunay
        # bistability.
        pinned={"sma": -3168.595092417778},
    ),
    _cli,
    _cell_args,
    __file__,
    _profiling_root(),
    _thread_env,
)
