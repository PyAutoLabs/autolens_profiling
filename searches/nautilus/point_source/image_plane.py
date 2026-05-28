"""First-class af.Nautilus search profiling — point-source image-plane fit.

Image-plane fit uses ``al.FitPositionsImagePairAll`` (chi-squared on the
solver-reproduced multiple-image positions).
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
    dataset_class="point_source",
    model_type="image_plane",
    default_instrument="simple",
)
