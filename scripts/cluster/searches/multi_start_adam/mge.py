"""af.MultiStartAdam search profiling — group MGE (4 lenses + 4 sources).

Fixed-step (``n_steps=300``) JAX/optax multi-start Adam gradient MAP optimizer on
the high-dimensional (~54 free-param) group model. Benchmarks whether a first-
order gradient optimizer scales to the 4-lens + 4-source cell and recovers the
input truth (scored against ``truth.json``). JAX-native — requires ``use_jax``.
See ``searches/README.md``.
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
    dataset_class="group",
    model_type="mge",
    default_instrument="hst",
)
