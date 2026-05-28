"""First-class af.NSS search profiling — datacube Delaunay.

Multi-channel cube fit via ``af.FactorGraphModel`` (same wiring as the
Nautilus datacube cell). ``_DATACUBE_N_CHANNELS`` in ``_setup.py``
controls the channel count.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="nss",
    dataset_class="datacube",
    model_type="delaunay",
    default_instrument="sma",
)
