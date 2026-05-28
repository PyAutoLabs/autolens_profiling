"""First-class af.NSS search profiling — interferometer Delaunay."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="nss",
    dataset_class="interferometer",
    model_type="delaunay",
    default_instrument="sma",
)
