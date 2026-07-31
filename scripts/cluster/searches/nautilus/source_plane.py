"""First-class af.Nautilus search profiling — cluster source-plane (free).

Multi-system cluster point-source fit via ``af.FactorGraphModel``, one
``al.AnalysisPoint`` per system. Plain free-centre source-plane fit
(``al.FitPositionsSource`` — un-weighted magnification chi-squared, Lenstool's
default likelihood; see ``scripts/cluster/likelihood_breakdown/source_plane.py``):
each source keeps a free ``al.ps.Point`` centre (uniform prior ±1.0" around
its own observed-positions centroid). Nautilus baseline for the #678 phase B
point-source-defaults evidence campaign at cluster scale — the un-weighted
counterpart to ``source_plane_tensor.py``'s jacobian-weighted variant.
"""

from __future__ import annotations

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


import sys
from pathlib import Path

_REPO_ROOT = _profiling_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="nautilus",
    dataset_class="cluster",
    model_type="source_plane",
    default_instrument="simple",
)
