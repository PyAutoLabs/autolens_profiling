"""First-class af.Nautilus search profiling — point-source image-plane solved.

Image-plane fit uses ``al.FitPositionsImagePairAllSolved`` (solver-reproduced
multiple-image positions with the source centre solved analytically — #657
phase 5). Nautilus anchor for the #678 phase B point-source-defaults evidence
campaign, and the discriminator arm run against the ``simple_missing`` /
``simple_extra`` instruments alongside ``image_plane_repeat_solved.py``.
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
    dataset_class="point_source",
    model_type="image_plane_solved",
    default_instrument="simple",
)
