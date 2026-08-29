"""First-class af.BlackJAXNUTS search profiling — imaging MGE.

The framework's first MCMC cell (PROGRAMME.md Phase 6, "initialized posterior
sampling"). Drives a full ``af.BlackJAXNUTS`` fit — vmapped multi-chain
gradient MCMC on mainline blackjax — on the same MGE lens + MGE source imaging
model every other ``imaging/mge/hst`` row in this framework fits, so a NUTS row
sits directly beside the Nautilus, NSS and MultiStart rows for the same target.

MGE is deliberately the first (and, for now, only) NUTS cell: it is the one
target whose posterior geometry is already characterised (the measured 269x
prior/posterior anisotropy and |r|=0.95 correlations behind H6.1), and it is
cheap enough that a warm-vs-cold A/B is a 45-minute probe rather than an
overnight block. Pixelized cells are explicitly out of scope until this one has
a measured step rate.

Both halves of the Phase 6 pre-requisite are driven from here via
``SEARCHES_NUTS_*`` (see ``searches/README.md``): start-point injection from a
previous fit (``SEARCHES_NUTS_WARM_FROM``, which never touches the model's
priors) and inverse-mass-matrix control (``SEARCHES_NUTS_MASS``).

See ``multi_start_prodigy/mge.py`` for the same model under a gradient MAP
optimizer — the natural source of a warm start — and ``searches/README.md`` for
the design.
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
    sampler="nuts",
    dataset_class="imaging",
    model_type="mge",
    default_instrument="hst",
)
