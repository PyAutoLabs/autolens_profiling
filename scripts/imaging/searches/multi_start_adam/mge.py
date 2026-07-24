"""First-class af.MultiStartAdam search profiling — imaging MGE.

Drives a full ``af.MultiStartAdam`` fit on an MGE lens + MGE source imaging model
across the canonical instruments (hst / euclid / jwst / ao). MultiStartAdam is a
JAX / optax multi-start gradient MAP optimizer; this is the cell where the search
is meaningful and benchmark-proven (the GPU MAP-optimizer benchmark recovered the
truth basin on the HST MGE lens likelihood). See ``searches/README.md`` for the
design and the sweep workflow.

Note: MultiStartAdam is JAX-native and requires ``use_jax=True`` (the sweep runs
JAX-on by default); a pure-NumPy config will raise.
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
    sampler="multi_start_adam",
    dataset_class="imaging",
    model_type="mge",
    default_instrument="hst",
)
