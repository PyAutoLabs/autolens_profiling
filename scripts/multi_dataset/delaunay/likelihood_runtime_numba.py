"""
Numba CPU Profiling: Delaunay Multi-Dataset Likelihood (Runtime, standard size)
===============================================================================

The per-evaluation runtime of the NumPy / numba multi-dataset likelihood a Nautilus worker runs on
CPU (likelihood-lifecycle 2f, autolens_profiling#475): every dataset on the numba
``apply_sparse_operator_cpu`` operator, ``AnalysisImaging(use_jax=False)``, over the preset's
iid instance stream (instance 0 a discarded warm-up); the headline is the warm iid median of the
summed call. Threads are pinned to 1 before numpy is imported.

The cell (decision D6): phase 2a's hst imaging standard cell (Delaunay 1250 + 30 edge points,
MGE lens light, free ``AdaptSplit``, production over-sampling and adapt image, iid draw 0) fitted to
``--n-datasets`` (default 2) datasets, dataset ``e`` holding hst's data and noise map times
``s_e`` (1, 0.5), as ``af.FactorGraphModel`` of one ``AnalysisImaging`` per dataset (the workspace
``multi_dataset/modeling.py`` configuration). The row carries ``resources``, ``fit.png`` (dataset 0)
and ``parity.json`` (the sum of 2a's eager NumPy dense reference per dataset, decision D8). The
shared driver is ``scripts/misc/likelihood_breakdown/multi_dataset_standard.py``.

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

from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
_thread_env = _pin_thread_env(1)

import os as _smoke_os  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

from likelihood_breakdown.multi_dataset_standard import run_numba  # noqa: E402

run_numba(_cli, _thread_env, script_file=__file__, profiling_root=_profiling_root())
