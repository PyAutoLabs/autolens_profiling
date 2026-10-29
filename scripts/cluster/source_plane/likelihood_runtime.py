"""
JAX Profiling: Cluster Source-Plane Likelihood (Runtime)
========================================================

The per-evaluation runtime of the cluster source-plane likelihood (likelihood-lifecycle 2e,
autolens_profiling#474): the solved source-plane chi-squared (``FitPositionsSourceSolved`` with
``PointSolved``, the workspace cluster search's fit; decision D3, human 2026-10-29): the observed
images are ray-traced back to each source's plane, weighted by the lensing Jacobian's precision
tensor (PyAutoLens bdb44a87c), and measured from the analytically solved centre beta*, which is
marginalised over. No PointSolver; ``--standard-size`` is recorded and changes nothing.

The cell is the standard cluster of ``dataset/cluster/simple`` (2 main dPIE + 10 scaling dPIE +
NFW host; 2 sources at z = 1 and 2), one ``AnalysisPoint(use_jax=True)`` per system summed the
way a ``FactorGraphModel`` sums its factors, JIT-compiled end to end from a model instance whose
free parameters are arguments. The row records lower / compile / first call / steady state, the
steady median, the vmap per call (the catalogue's JAX headline), the plain fit class
(``FitPositionsSource``, free ``Point`` centres) as a sub-row, ``resources``, ``parity.json`` (JIT vs eager NumPy) and ``fit.png``. The shared
driver is ``scripts/misc/likelihood_breakdown/cluster_standard.py`` (``run_runtime``).

Output: the mirrored cell named by ``--output-path`` (the campaign submits always pass one),
else ``results/runtime/cluster/source_plane/...`` placed by ``_profile_cli``.
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

run_runtime("source_plane", parse_profile_cli(), profiling_root=_profiling_root())
