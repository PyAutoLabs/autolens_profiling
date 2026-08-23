"""First-class af.MultiStartProdigy search profiling — imaging rectangular pixelized source.

Promoted from the autolens_workspace_developer#117 campaign (2026-07,
``searches_minimal/pix_prodigy_findings.md``). The rectangular kernel-CDF
family is the C-infinity end of the mesh-smoothness spectrum — in the
campaign its trajectories were smooth, steady, minimal-churn climbs (it
passed the #101 adam-era 3000-step endpoint within 25 Prodigy steps).

STATUS CAVEAT (2026-07-28): the campaign's rectangular verdict is
**throughput-limited, not landscape-limited** — on 32 CPUs its
``value_and_grad`` step cost ~5.7 min (~17x the knn cell, well above its
~4.5x forward-eval ratio, i.e. the kernel-CDF jvp is disproportionately
expensive), so its broad-start chain had not yet reached the step count at
which the other meshes' late breakouts occurred. Nothing in its trajectory
resembles the AdaptSplit wall signature. Profiling that jvp cost on the A100
is the listed follow-up; this cell is the vehicle.

Budget/batching lessons as per the ``knn`` cell: 16 starts, 3000-step budget
(a long plateau is a reg mode, not convergence), ``batch_size=4`` mandatory
(unbatched 16-start pixelized jvp ~58 GB). Model: MGE lens light + free broad
Isothermal/shear + ``RectangularRTUAdaptImage`` source + free ``Constant`` reg —
the SLaM ``source_pix[1]`` shape (see ``_setup._pixelization_model``).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)


import sys
from pathlib import Path

_REPO_ROOT = _profiling_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="multi_start_prodigy",
    dataset_class="imaging",
    model_type="pixelization",
    default_instrument="hst",
)
