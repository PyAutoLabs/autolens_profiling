"""af.MultiStartLion search profiling — group MGE (4 lenses + 4 sources).

Fixed-step (``n_steps=300``) JAX multi-start Lion gradient MAP optimizer on the
high-dimensional (~54 free-param) group model. JAX-native — requires ``use_jax``.
See ``searches/README.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # autolens_profiling/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="multi_start_lion",
    dataset_class="group",
    model_type="mge",
    default_instrument="hst",
)
