"""af.MultiStartProdigy (auto-convergence) search profiling — group MGE.

The recently-shipped **automatic-convergence** mode: multi-start Prodigy wrapped
in ``af.MultiStartGradientConvergence`` so each start early-stops when its
figure-of-merit plateaus, instead of always running the fixed ``n_steps=300``.
Benchmarked head-to-head with the fixed-step ``multi_start_prodigy`` cell on the
same 4-lens + 4-source model — the honest "how few steps does it actually need"
measure. The summary records the convergence criterion. JAX-native — requires
``use_jax``. See ``searches/README.md``.
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
    sampler="multi_start_prodigy_autoconv",
    dataset_class="group",
    model_type="mge",
    default_instrument="hst",
)
