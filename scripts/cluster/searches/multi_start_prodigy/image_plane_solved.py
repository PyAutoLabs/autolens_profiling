"""First-class af.MultiStartProdigy search profiling — cluster image-plane solved.

Multi-system cluster point-source fit via ``af.FactorGraphModel``, one
``al.AnalysisPoint`` per system. Gradient search on the solved image-plane
likelihood (``al.FitPositionsImagePairRepeatSolved`` — forward-solves the
lens equation per source via the ``PointSolver``, source centre solved
analytically). JAX-only; the PointSolver implicit-diff ``custom_jvp``
(#657 phase 5) is what makes this gradient-searchable at all. Prodigy anchor
for the #678 phase B point-source-defaults evidence campaign at cluster
scale. As with the Nautilus sibling, there is deliberately no free-centre
cluster image_plane cell — the ~0.3 s/call forward solve makes sampling a
free source centre wall-time-prohibitive at cluster scale.
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
    sampler="multi_start_prodigy",
    dataset_class="cluster",
    model_type="image_plane_solved",
    default_instrument="simple",
)
