"""
Numba CPU Profiling: Delaunay Datacube Likelihood (Runtime, standard size)
==========================================================================

The per-evaluation runtime of the NumPy / numba datacube likelihood a Nautilus worker runs on CPU
(likelihood-lifecycle 2f, autolens_profiling#475): ``af.FactorGraphModel`` over one
``AnalysisInterferometer(use_jax=False, shared_preloads=True)`` per channel, each on a dataset
with the NumPy W~ sparse operator, so the library routes every evaluation itself. The cube is the
2b interferometer standard cell in every channel (decision D2, ``--n-channels``, default 34),
channel visibilities scaled by an emission line; the lead channel builds the mapper and the
curvature matrix once and every channel reuses them.

The headline is the packaged cost-model route (2b ruling R4) over the preset's iid stream, its
warm median per cube evaluation (``--arms`` adds the forced ``numba`` / ``numpy_fft`` routes
as sub-rows; off by default, decision D5). Threads are pinned to 1 before numpy is imported. The
row carries ``resources``, ``fit.png`` (the lead channel) and ``parity.json`` (the cube
reference, decision D3). The shared driver is
``scripts/misc/likelihood_breakdown/datacube_standard.py`` (``run_numba``).

Output: the mirrored cell named by ``--output-path`` (the campaign submits always pass one).
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
from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
_thread_env = _pin_thread_env(1)

import os as _smoke_os  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

from likelihood_breakdown.datacube_standard import run_numba  # noqa: E402

run_numba(_cli, _thread_env, script_file=__file__, profiling_root=_profiling_root())
