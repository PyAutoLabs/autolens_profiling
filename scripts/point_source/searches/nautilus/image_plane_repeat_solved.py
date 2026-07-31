"""First-class af.Nautilus search profiling — point-source image-plane repeat solved.

Image-plane fit uses ``al.FitPositionsImagePairRepeatSolved`` (repeat-pairing
image-plane chi-squared with the source centre solved analytically): the
discriminator model type for the #678 phase B point-source-defaults evidence
campaign, run against ``simple``, ``simple_missing`` (a dropped image) and
``simple_extra`` (a spurious extra position) to test robustness to an
incorrect image count versus ``image_plane_solved.py``.
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
    model_type="image_plane_repeat_solved",
    default_instrument="simple",
)
