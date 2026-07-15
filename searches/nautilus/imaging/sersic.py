"""First-class af.Nautilus search profiling — imaging Sersic.

The Nautilus **baseline** for the optimizer-tuning campaign
(autolens_profiling#69): the gradient MAP optimizers' settings are only
meaningful measured against the sampler users run today, on the identical model
(``searches/_setup.py::_sersic_model``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # autolens_profiling/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="nautilus",
    dataset_class="imaging",
    model_type="sersic",
    default_instrument="hst",
)
