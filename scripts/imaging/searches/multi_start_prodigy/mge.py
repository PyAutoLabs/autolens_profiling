"""First-class af.MultiStartProdigy search profiling — imaging MGE.

Added for W5 Phase 8B (issue #162): the MGE control cell for the bijector A/B
(``bijector_ab.py``). MGE has no pixelization / regularization coefficients at
all, so it carries no ``LogUniformPrior`` under a ``"regularization."`` path —
the ``log_reg`` bijector arm (``searches._samplers._bijector_object``)
therefore resolves to an empty ``kind_by_path`` and is bit-identical to
``BijectorNone`` here. That is the point: MGE is the falsification control for
F4 ("MGE control differs by any bit"), run alongside the pixelized cells
rather than through a bespoke path.

See ``multi_start_adam/mge.py`` for the same model under a different
optimizer, and ``searches/README.md`` for the design.
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
    sampler="multi_start_prodigy",
    dataset_class="imaging",
    model_type="mge",
    default_instrument="hst",
)
