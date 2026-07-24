"""First-class af.Nautilus search profiling — group MGE (4 lenses + 4 sources).

Drives a full ``af.Nautilus`` fit on the high-dimensional (~54 free-param)
group-scale model: 4 deflectors (MGE light + Isothermal mass) lensing 4 MGE
sources. This is the **reference / anchor** for the group benchmark — if
Nautilus cannot recover the input truth here, the simulation or model is wrong
and the JAX gradient optimizers should not be trusted on the same cell. See
``searches/README.md`` for design and the sweep workflow.
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

_REPO_ROOT = _profiling_root()  # autolens_profiling/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="nautilus",
    dataset_class="group",
    model_type="mge",
    default_instrument="hst",
)
