"""First-class af.Nautilus search profiling — datacube Delaunay.

Multi-channel cube fit via ``af.FactorGraphModel``: N identical channel
datasets, each wrapped in ``al.AnalysisInterferometer`` + ``af.AnalysisFactor``,
combined under a single global model — mirrors
``autolens_workspace/scripts/multi/modeling.py``. The channel count comes
from ``_DATACUBE_N_CHANNELS`` in ``searches/_setup.py`` (default 4 to
match the existing ``likelihood_runtime/datacube/delaunay.py`` quick-
iteration value).
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
    dataset_class="datacube",
    model_type="delaunay",
    default_instrument="sma",
)
