"""First-class af.MultiStartAdam search profiling — imaging MGE.

Drives a full ``af.MultiStartAdam`` fit on an MGE lens + MGE source imaging model
across the canonical instruments (hst / euclid / jwst / ao). MultiStartAdam is a
JAX / optax multi-start gradient MAP optimizer; this is the cell where the search
is meaningful and benchmark-proven (the GPU MAP-optimizer benchmark recovered the
truth basin on the HST MGE lens likelihood). See ``searches/README.md`` for the
design and the sweep workflow.

Note: MultiStartAdam is JAX-native and requires ``use_jax=True`` (the sweep runs
JAX-on by default); a pure-NumPy config will raise.
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
    model_type="mge",
    default_instrument="hst",
)
