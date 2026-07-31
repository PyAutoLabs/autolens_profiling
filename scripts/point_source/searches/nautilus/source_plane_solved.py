"""First-class af.Nautilus search profiling — point-source source-plane solved.

Source-plane fit uses ``al.FitPositionsSourceSolved`` (the source centre is
solved analytically rather than sampled — #657 phase 5). Nautilus reference
anchor for the #678 phase B point-source-defaults evidence campaign: feeds the
truth-anchored ``truth_log_likelihood`` / ``delta_max_ll_vs_truth`` and
``posterior_stats`` recommendation for the SLaM point-source default.
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
    model_type="source_plane_solved",
    default_instrument="simple",
)
