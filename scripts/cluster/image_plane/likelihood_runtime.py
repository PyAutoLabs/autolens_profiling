"""
JAX Profiling: Cluster Image-Plane Likelihood (Runtime)
=======================================================

The per-evaluation runtime of the cluster image-plane likelihood (likelihood-lifecycle 2e,
autolens_profiling#474): the solved image-plane chi-squared (``FitPositionsImagePairRepeatSolved``
with ``PointSolved``, decision D3, human 2026-10-29): each source's centre is solved analytically
(beta*, the tensor-weighted mean of its back-traced images), its images are forward-solved with the
``PointSolver`` (``--standard-size``: the workspace cluster solver, 100 x 100 @ 1.0", precision
0.001, ``residual_threshold`` 10, decision D1) and paired with the observed ones.

The cell is the standard cluster of ``dataset/cluster/simple`` (2 main dPIE + 10 scaling dPIE +
NFW host; 2 sources at z = 1 and 2), one ``AnalysisPoint(use_jax=True)`` per system summed the
way a ``FactorGraphModel`` sums its factors, JIT-compiled end to end from a model instance whose
free parameters are arguments. The row records lower / compile / first call / steady state, the
steady median, the vmap per call (the catalogue's JAX headline), the plain fit class
(``FitPositionsImagePairRepeat``, free ``Point`` centres) as a sub-row, ``resources``, ``parity.json`` (JIT vs eager NumPy) and ``fit.png``. The shared
driver is ``scripts/misc/likelihood_breakdown/cluster_standard.py`` (``run_runtime``).

Output: the mirrored cell named by ``--output-path`` (the campaign submits always pass one),
else ``results/runtime/cluster/image_plane/...`` placed by ``_profile_cli``.
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

import os as _smoke_os  # noqa: E402

import autolens  # noqa: E402, F401
import jax  # noqa: E402, F401

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

from likelihood_breakdown.cluster_standard import run_runtime  # noqa: E402

from _profile_cli import parse_profile_cli  # noqa: E402

run_runtime("image_plane", parse_profile_cli(), profiling_root=_profiling_root())
