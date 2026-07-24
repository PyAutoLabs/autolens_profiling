"""First-class af.Nautilus search profiling — group MGE (4 lenses + 4 sources).

Drives a full ``af.Nautilus`` fit on the high-dimensional (~54 free-param)
group-scale model: 4 deflectors (MGE light + Isothermal mass) lensing 4 MGE
sources. This is the **reference / anchor** for the group benchmark — if
Nautilus cannot recover the input truth here, the simulation or model is wrong
and the JAX gradient optimizers should not be trusted on the same cell. See
``searches/README.md`` for design and the sweep workflow.
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
    dataset_class="group",
    model_type="mge",
    default_instrument="hst",
)
