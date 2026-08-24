"""First-class af.Nautilus search profiling — imaging MGE lens + RectangularRTUAdaptImage source (free reg.Adapt) — production-SLaM-shaped SOURCE-PIX target.

Drives a full ``af.Nautilus`` fit on the MGE lens + RectangularRTUAdaptImage source (free reg.Adapt) — production-SLaM-shaped SOURCE-PIX target imaging model
(``searches._setup._slam_source_pix_model``). See ``searches/README.md``
for design and the sweep workflow, and ``results/notes/inference/
DECISIONS.md`` (2026-08-24, W4 / issue #161) for why this target is
registered.
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
    dataset_class="imaging",
    model_type="slam_source_pix",
    default_instrument="hst",
)
