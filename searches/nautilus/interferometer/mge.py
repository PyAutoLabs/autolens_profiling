"""First-class af.Nautilus search profiling — interferometer MGE.

Drives a full ``af.Nautilus`` fit on an MGE lens + MGE source interferometer
model across the canonical instruments (sma / alma / alma_high / jvla).
See ``searches/README.md`` for the sweep workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="nautilus",
    dataset_class="interferometer",
    model_type="mge",
    default_instrument="sma",
)
