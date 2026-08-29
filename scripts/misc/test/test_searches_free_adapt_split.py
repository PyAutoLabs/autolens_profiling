"""Unit tests for the corrected free-adapt-split target (issue #196).

WHAT THESE PIN, AND WHY
-----------------------
RAL 341908_5 was killed by a likelihood-overflow flood: ``al.reg.AdaptSplit``
squares its coefficient twice, so under the legacy ``LogUniform(1e-6, 1e6)``
prior the regularization term reaches 1e24, the matrix goes non-PD from
c ~ 1e4, and the fp64 Cholesky returns FINITE GARBAGE — ``log_l`` up to 3e+303,
accepted by Nautilus as the best point (DECISIONS.md 2026-08-29).

``searches/_setup._free_adapt_split`` is the fix, and it has three properties a
future edit could silently break:

1. **The class.** ``al.reg.AdaptSplitPower``, not ``AdaptSplit`` — squared once.
2. **The cap.** ``LogUniform(1e-6, 1e4)``, not ``1e6``. The class change alone
   still admits c^2 = 1e12 at the top of the legacy prior; the cap is what keeps
   the sampler off the non-PD region.
3. **The model dimension is unchanged.** ``power`` is a ``Constant``, never
   sampled. If it ever became a free prior, every ``model_dim`` in the registry
   would move and the corrected rows would not be comparable with anything.

Plus the documented identity the helper exists to guarantee: ``knn`` and
``delaunay_adapt_split`` are parameter-identical and differ only in the mesh
(``_setup._delaunay_adapt_split_model``'s docstring), which is only true while
both build their regularization through the one helper.

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_free_adapt_split.py -q
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
for _p in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import pytest  # noqa: E402
from searches._setup import _build_model, _free_adapt_split  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_FREE_ADAPT_SPLIT_CELLS = ("knn", "delaunay_adapt_split", "slam_source_pix_nn")
_MASK_RADIUS = 3.0


def _regularization(model_type: str):
    model = _build_model("imaging", model_type, mask_radius=_MASK_RADIUS)
    return model, model.galaxies.source.pixelization.regularization


@pytest.mark.parametrize("model_type", _FREE_ADAPT_SPLIT_CELLS)
def test_cell_uses_the_squared_once_class(model_type):
    _, reg = _regularization(model_type)
    assert reg.cls is al.reg.AdaptSplitPower


@pytest.mark.parametrize("model_type", _FREE_ADAPT_SPLIT_CELLS)
def test_coefficient_priors_are_capped_below_the_non_pd_onset(model_type):
    """1e4 is where the matrix was measured to go non-PD, not a round number."""
    _, reg = _regularization(model_type)
    for name in ("inner_coefficient", "outer_coefficient"):
        prior = getattr(reg, name)
        assert isinstance(prior, af.LogUniformPrior), name
        assert prior.lower_limit == pytest.approx(1e-6), name
        assert prior.upper_limit == pytest.approx(1e4), name


@pytest.mark.parametrize("model_type", _FREE_ADAPT_SPLIT_CELLS)
def test_power_is_a_constant_and_never_sampled(model_type):
    """If `power` ever became free, every registry model_dim would move."""
    model, reg = _regularization(model_type)
    assert reg.power == pytest.approx(1.0)
    assert "power" not in [
        name.split(".")[-1] for name in model.model_component_and_parameter_names
    ]


def test_knn_and_delaunay_adapt_split_stay_parameter_identical():
    """The documented three-way comparison isolates mesh from reg only while
    these two differ in nothing but the mesh."""
    knn, knn_reg = _regularization("knn")
    das, das_reg = _regularization("delaunay_adapt_split")
    assert knn.prior_count == das.prior_count
    assert knn.model_component_and_parameter_names == das.model_component_and_parameter_names
    assert knn_reg.cls is das_reg.cls
    # The meshes are pinned INSTANCES, not Models (see _delaunay_model's note on
    # why the bare class form cannot be used), so this is an isinstance check.
    assert isinstance(knn.galaxies.source.pixelization.mesh, al.mesh.KNearestNeighbor)
    assert isinstance(das.galaxies.source.pixelization.mesh, al.mesh.Delaunay)


def test_signal_scale_is_pinned_on_knn_and_free_on_slam_source_pix_nn():
    """Deliberate and out of scope to change (#196): the two cells differ here,
    and `slam_source_pix_nn`'s extra free parameter is why it is 14-D not 13-D."""
    knn, knn_reg = _regularization("knn")
    sspnn, sspnn_reg = _regularization("slam_source_pix_nn")
    assert knn_reg.signal_scale == pytest.approx(1.0)
    assert isinstance(sspnn_reg.signal_scale, af.UniformPrior)
    assert sspnn.prior_count == knn.prior_count + 1


def test_helper_can_leave_signal_scale_free():
    pinned = _free_adapt_split(signal_scale=1.0)
    free = _free_adapt_split(signal_scale=None)
    assert pinned.prior_count == free.prior_count - 1


def test_helper_cap_is_honoured():
    reg = _free_adapt_split(cap=1e2)
    assert reg.inner_coefficient.upper_limit == pytest.approx(1e2)
    assert reg.outer_coefficient.upper_limit == pytest.approx(1e2)
