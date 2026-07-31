"""First-class af.Nautilus search profiling — cluster image-plane solved.

Multi-system cluster point-source fit via ``af.FactorGraphModel``, one
``al.AnalysisPoint`` per system. Image-plane fit uses
``al.FitPositionsImagePairRepeatSolved`` (the cluster "model-fit default"
per ``scripts/cluster/likelihood_breakdown/image_plane.py`` — forward-solves
the lens equation per source via the ``PointSolver``, with the source centre
solved analytically). This is the validation-heavy end of the campaign: the
forward solve costs ~0.3 s/call at this repo's solver grid (200x200 @ 0.7",
precision 0.01"), so there is deliberately **no free-centre cluster
image_plane cell** — sampling a free source centre through a per-evaluation
forward solve is wall-time-prohibitive at cluster scale (#678 phase B).
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
    model_type="image_plane_solved",
    default_instrument="simple",
)
