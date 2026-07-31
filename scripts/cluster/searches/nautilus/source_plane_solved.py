"""First-class af.Nautilus search profiling — cluster source-plane solved.

Multi-system cluster point-source fit (2 main dPIE lenses + scaling tier +
NFW host halo + N sources) via ``af.FactorGraphModel``, one
``al.AnalysisPoint`` per system. Source-plane fit uses
``al.FitPositionsSourceSolved`` (each source's centre solved analytically —
#657 phase 5) — the recommended cluster search-stage fit (see
``autolens_workspace/scripts/cluster/modeling.py``). Nautilus reference
anchor for the #678 phase B point-source-defaults evidence campaign at
cluster scale: feeds the truth-anchored ``truth_log_likelihood`` /
``delta_max_ll_vs_truth`` and ``posterior_stats``.
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
    model_type="source_plane_solved",
    default_instrument="simple",
)
