"""
Numba CPU Profiling: Delaunay Interferometer Likelihood (Runtime, standard size)
================================================================================

The per-evaluation runtime of the NumPy / numba interferometer likelihood a Nautilus worker runs
on CPU (likelihood-lifecycle 2b, autolens_profiling#469): ``AnalysisInterferometer(use_jax=False)``
on a dataset with the NumPy W~ sparse operator, so the library routes every evaluation itself.
The cell is the standard-size interferometer production cell (``--standard-size``, ruling R3;
``_production_config.interferometer_standard_preset``), shared with every other setup of the
likelihood.

Arms (ruling R4), one iid instance stream, threads pinned to 1:

- ``default`` -- the headline: the packaged route, the cost model of PyAutoArray#628
  (``Settings.interferometer_numba_nnz_per_source_max=None``), which picks numba ``direct_conv``
  or the NumPy FFT route per evaluation; the route it took is recorded;
- ``numba`` / ``numpy_fft`` -- the two routes forced (``Settings(interferometer_route=...)``),
  written as sub-rows under ``arms``, never the headline.

Instance 0 is a discarded warm-up (numba compile), the next ``--cold-evals`` are cold, the rest
warm; the headline is the default arm's warm iid median. ``--n-instances`` is 20 / 20 / 10 / 5
on sma / alma / alma_high / jvla in the campaign submits (R4). The row carries ``resources``,
``fit.png`` and ``parity.json`` (the chunked dense reference, R6 / R7). The shared driver is
``scripts/misc/likelihood_breakdown/interferometer_standard.py`` (``run_numba``).

Output: the mirrored cell named by ``--output-path`` (the campaign submits always pass one),
else ``results/runtime/interferometer/delaunay_numba/...`` placed by ``_profile_cli``.
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

from likelihood_breakdown.interferometer_standard import run_numba  # noqa: E402

run_numba("delaunay", _cli, _thread_env, script_file=__file__, profiling_root=_profiling_root())
