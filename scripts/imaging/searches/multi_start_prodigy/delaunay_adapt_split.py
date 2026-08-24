"""First-class af.MultiStartProdigy search profiling — imaging Delaunay + free
AdaptSplit regularization.

Added for W5 Phase 8B (issue #162): the NaN-wall cell for the bijector A/B
(``bijector_ab.py``). Unlike ``delaunay.py`` (``model_type="delaunay_matern"``,
the REGISTERED gradient cell — free Matérn regularization, which degrades
gracefully), this uses ``model_type="delaunay_adapt_split"``
(``searches._setup._delaunay_adapt_split_model``): Delaunay mesh + free
``AdaptSplit`` regularization, the pairing Phase 8A (CP-4, ``slogdet_ab.py``)
identified as the actual **NaN wall** — the #104 doubly-squared lambda^4
fragility, lanes dying instead of learning at high coefficients. It is a
DIAGNOSTIC cell, not a sampler-benchmark recommendation (see
``_setup._delaunay_adapt_split_model``'s docstring): it exists so the bijector
A/B can be run against the configuration that actually walls, alongside
``knn`` (the cell the original pre-registration named, whose high-coefficient
region is a finite floor rather than a wall).
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
    model_type="delaunay_adapt_split",
    default_instrument="hst",
)
