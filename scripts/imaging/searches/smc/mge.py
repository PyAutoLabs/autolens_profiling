"""First-class af.SMC search profiling — imaging MGE.

The framework's second MCMC cell (PROGRAMME.md Phase 7, "particle / sequential
Monte Carlo") and the first sampler here that produces a posterior AND a
``log_evidence`` by a gradient route. Drives a full ``af.SMC`` fit — blackjax
adaptive-tempered SMC with a MALA or HMC rejuvenation kernel — on the same MGE
lens + MGE source imaging model every other ``imaging/mge/hst`` row in this
framework fits, so an SMC row sits directly beside the Nautilus, NSS, NUTS and
MultiStart rows for the same target.

WHY THIS CELL, AND WHY IT IS THE ONE WORTH RUNNING NEXT
------------------------------------------------------
Nautilus's ``log_evidence`` is this programme's incumbent bar, and until now
nothing in the framework could produce an independent estimate of the same
quantity: NSS is another nested sampler, NUTS and the MultiStart optimizers give
no evidence at all. SMC's tempering bridge does — by a completely different
route, from the gradient side — so a cold SMC run is the first available
cross-check on the number every gate in this programme is scored against. That
is what the ``mala_cold`` arm of the A100 probe is for.

MGE is deliberately the first (and, for now, only) SMC cell, for the reasons the
NUTS leaf gives: its posterior geometry is already characterised (the measured
269x prior/posterior anisotropy and |r| = 0.95 correlations behind H6.1), and it
is cheap enough that a three-arm probe is a couple of GPU-hours rather than an
overnight block. Pixelized cells are out of scope until this one has a measured
step rate — the substitution that killed RAL job 340576.

READ THE SCHEDULE, NOT THE FLAG
-------------------------------
``af.SMC`` reports ``converged`` when the tempering parameter reaches 1.0 and
nothing else, and an adaptive schedule will walk a collapsed particle cloud all
the way there. Every run from this leaf therefore carries a ``diagnostics``
block with the lambda schedule, the per-step acceptance trace and the per-step
ESS (``searches/_runner._smc_diagnostics``), and it marks itself ``valid: false``
on a partial tempering path or a collapsed cloud. A "converged" SMC row whose
final ESS is 3 particles is not a measurement of anything.

Driven from ``SEARCHES_SMC_*`` (see ``searches/README.md``): particle count,
inner kernel, rejuvenation steps, target ESS, and the warm-start /
whitening arm.

See ``nuts/mge.py`` for the same model under gradient MCMC and
``multi_start_prodigy/mge.py`` for it under a gradient MAP optimizer — the
natural source of a warm start — and ``searches/README.md`` for the design.
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
    sampler="smc",
    dataset_class="imaging",
    model_type="mge",
    default_instrument="hst",
)
