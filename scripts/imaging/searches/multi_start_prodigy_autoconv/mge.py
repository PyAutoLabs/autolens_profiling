"""af.MultiStartProdigy (auto-convergence) search profiling — imaging MGE.

The CP-3 cell of the inference programme (``results/notes/inference/
PROGRAMME.md`` §4 Phase 3, §9 CP-3): Prodigy's per-start **basin-hit
probability** on ``imaging/mge/hst``, measured by per-lane basin
classification across ``n_starts`` x seeds rather than by accumulating
whole-run anecdotes (§2.3).

Auto-convergence is ON — ``n_steps`` is a ceiling and the run early-stops when
the global-best figure of merit plateaus (window 50, rtol 1e-4, atol 1e-3, min
100 steps), with ``stop_reason`` persisted. Phase 3 lists termination itself as
a benchmark metric (§3), and its convergence-detector confusion matrix
(stopped-correct / stopped-wrong-basin / ceiling) is read off that field.

Wave 1 is **positions-off**. The positions-on half of CP-3 needs PositionsLH
plumbing this framework does not have (``_setup.py`` builds no
``al.AnalysisImaging(positions_likelihood=...)``), and it is deliberately out
of scope here.

Requires **PyAutoFit >= PR#1515** (``feature/multistart-per-lane-best``): the
per-lane best records (``lane_best_params`` / ``lane_best_foms`` /
``lane_best_steps``) it adds to ``search_internal`` are what per-lane basin
classification is computed from. Against an older PyAutoFit the run still
completes, but its ``diagnostics`` block reports ``valid: false`` with that
reason attached.

Every knob is an env override (see ``searches/_samplers.py``); the CP-3 arms
set them from ``hpc/batch_gpu/submit_search_multi_start_prodigy_autoconv_*``::

    SEARCHES_N_STARTS=64        # arm: {16, 64, 256}
    SEARCHES_N_STEPS=3000       # ceiling, matching the recorded Prodigy MGE budget
    SEARCHES_SEED=0             # >= 5 seeds per arm; also drives the unique_tag
    SEARCHES_CLIPPER=prior_box  # best-supported config (hygiene)
    SEARCHES_SCALER=none        # falsified as a fix; not an arm here
    SEARCHES_DISABLE_VIZ=1      # viz is not the measurement

Note the un-overridden defaults for this cell are the framework's generic
``n_starts=64`` / ``n_steps=300``; a 300-step read of MGE Prodigy would be a
budget artefact (the recorded 16x3000 rows stop on ``max_steps``), so an arm
that does not set ``SEARCHES_N_STEPS`` is not a CP-3 arm.
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

_REPO_ROOT = _profiling_root()  # autolens_profiling/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._runner import run_search  # noqa: E402

run_search(
    sampler="multi_start_prodigy_autoconv",
    dataset_class="imaging",
    model_type="mge",
    default_instrument="hst",
)
