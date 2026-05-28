"""First-class af.Nautilus search profiling — datacube Delaunay.

Multi-channel cube fit via ``af.FactorGraphModel``: N identical channel
datasets, each wrapped in ``al.AnalysisInterferometer`` + ``af.AnalysisFactor``,
combined under a single global model — mirrors
``autolens_workspace/scripts/multi/modeling.py``. The channel count comes
from ``_DATACUBE_N_CHANNELS`` in ``searches/_setup.py`` (default 4 to
match the existing ``likelihood_runtime/datacube/delaunay.py`` quick-
iteration value).
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
    dataset_class="datacube",
    model_type="delaunay",
    default_instrument="sma",
)
