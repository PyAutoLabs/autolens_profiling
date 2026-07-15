"""First-class af.MultiStartAdam search profiling — imaging Sersic.

The lowest-complexity parametric cell: a Sersic lens + Sersic source (see
``searches/_setup.py::_sersic_model``). Together with ``imaging/mge`` this is the
model-complexity axis of the optimizer-tuning campaign (autolens_profiling#69),
which tunes ``n_starts`` / ``n_steps`` / ``learning_rate`` against a Nautilus
baseline so the JAX gradient MAP optimizers can become a standard option for
users with parametric sources.

Settings come from ``_samplers.multi_start_settings()``; override per grid point
with e.g. ``MULTI_START_SETTINGS="n_starts=32,n_steps=1000,learning_rate=0.001"``
and a matching ``--config-name`` so each point writes its own artifact.

MultiStartAdam is JAX-native; the sweep runs JAX-on by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # autolens_profiling/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="multi_start_adam",
    dataset_class="imaging",
    model_type="sersic",
    default_instrument="hst",
)
