"""Sampler factories for ``searches/``.

A small registry that maps sampler name → factory function. Every factory
returns a first-class PyAutoFit search object (``af.Nautilus`` today,
``af.DynestyStatic`` / ``af.Emcee`` / ``af.BlackJAXNUTS`` / ... in future).

The runner imports ``SAMPLER_BUILDERS`` and dispatches without per-sampler
branching elsewhere. Adding a new sampler is a single function + one dict
row.

The per-(dataset_class, model_type) ``n_live`` values mirror the SLaM
pipeline canonical settings in
``autolens_workspace/scripts/guides/modeling/slam_start_here.py`` —
``source_lp[1]`` uses ``n_live=200`` (MGE / parametric sources) and
``source_pix[1]`` uses ``n_live=150`` (pixelization / Delaunay). Point-
source phases are parametric like ``source_lp[1]`` so use 200; datacube
Delaunay matches imaging Delaunay at 150.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import autofit as af

# ``vram/config.py`` lives at the workspace root and stores per-(dataset, model,
# instrument) A100-probed vmap batch sizes. The samplers read it so we don't
# hardcode batch sizes that would OOM on heavier cells (e.g. imaging/delaunay
# at HST scale uses batch=16, not the Nautilus-default 100).
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
from vram import vmap_batch_for  # noqa: E402

# (dataset_class, model_type) -> n_live. Matches the SLaM defaults so a
# profiling row is comparable to a real source phase.
_N_LIVE: dict[tuple[str, str], int] = {
    ("imaging", "mge"): 200,
    ("group", "mge"): 200,
    ("imaging", "pixelization"): 150,
    ("imaging", "delaunay"): 150,
    ("interferometer", "mge"): 200,
    ("interferometer", "pixelization"): 150,
    ("interferometer", "delaunay"): 150,
    ("point_source", "image_plane"): 200,
    ("point_source", "source_plane"): 200,
    ("datacube", "delaunay"): 150,
}


def n_live_for(dataset_class: str, model_type: str) -> int:
    """Look up the canonical n_live for a (dataset_class, model_type) cell."""
    try:
        return _N_LIVE[(dataset_class, model_type)]
    except KeyError as exc:
        raise KeyError(
            f"No n_live preset for ({dataset_class!r}, {model_type!r}). "
            f"Add a row to ``_N_LIVE`` in ``searches/_samplers.py``."
        ) from exc


# Cells the vram probe hasn't covered fall back to the Nautilus default of 100.
# Probed cells (everything under (imaging, *, *) and (interferometer, *, *)) use
# the A100-validated value so we don't OOM on inversion-heavy cells like
# imaging/delaunay × hst (922 MB / replica → batch=16 max).
_FALLBACK_BATCH = 100


def vmap_batch_for_cell(dataset_class: str, model_type: str, instrument: str) -> int:
    """Resolve the per-cell vmap batch size from the vram registry.

    Returns ``vram.vmap_batch_for(...)`` when probed; ``_FALLBACK_BATCH`` for
    point_source / datacube / un-probed cells (these have small inversions
    or no vmap surface and the Nautilus default is fine).
    """
    val = vmap_batch_for(dataset_class, model_type, instrument)
    return val if val is not None else _FALLBACK_BATCH


def build_nautilus(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    config_name: str,
    use_jax: bool,
) -> af.Nautilus:
    """Construct a first-class ``af.Nautilus`` search for one profiling cell.

    Profiling-specific choices:

    - ``number_of_cores=1`` for every config so what's measured is per-
      evaluation cost, not pool throughput. Production scaling via
      ``number_of_cores > 1`` is a separate sweep axis.
    - ``force_x1_cpu=use_jax`` because ``nautilus.Sampler`` would fork a
      multiprocessing pool and corrupt JAX state otherwise.
    - ``use_jax_vmap=use_jax`` so JAX rows get the batched-evaluation
      win and NumPy rows get the standard per-sample path.
    - ``force_pickle_overwrite=True`` so output pickle files in the
      ``files/`` directory get re-written when an existing search is
      re-touched (useful when code that produces them has changed).
      NOTE: this does **not** bypass the ``.completed`` resume gate —
      that's handled at the sweep level (see ``sweep.py``'s
      ``--keep-completed`` flag; the default wipes search state).
    - ``iterations_per_update`` set explicitly so the visualization
      cadence does not silently change across PyAutoFit versions.
    """
    n_live = n_live_for(dataset_class, model_type)
    n_batch = vmap_batch_for_cell(dataset_class, model_type, instrument)
    return af.Nautilus(
        name=config_name,
        path_prefix=f"searches/{sampler}/{dataset_class}/{model_type}/{instrument}",
        n_live=n_live,
        n_batch=n_batch,
        number_of_cores=1,
        force_x1_cpu=use_jax,
        use_jax_vmap=use_jax,
        force_pickle_overwrite=True,
        iterations_per_update=3 * n_live,
    )


# MultiStart profiling settings. Single-sourced here so the builder and the
# JSON config block (``_runner._sampler_config_dict``) record identical values.
# These are illustrative profiling values, not the A100 scaling run (the
# GIGA-Lens recipe uses hundreds of starts); ``n_starts=64`` is a representative
# multi-start batch for a local/A100 profile.
_MULTI_START_N_STARTS = 64
_MULTI_START_N_STEPS = 300
_MULTI_START_LEARNING_RATE = 0.01

# Per-dataset-class ``n_starts`` overrides, driven by VRAM. Each start is a
# vmap replica of the whole likelihood, so the group cell (4 lenses + 4 sources
# = 54 params, 8 MGE bases through one inversion) has a much larger replica than
# the single-lens cell. Measured: 64 starts requests a single 4.71 GiB
# allocation and OOMs a 6 GB laptop GPU (RTX 2060, capped at 50%). An A100
# (80 GB) runs the full 64 comfortably — so this is a *local* accommodation,
# not a statement about the method. ``SEARCHES_N_STARTS`` overrides either way
# (set it back to 64 for the A100 rows).
_MULTI_START_N_STARTS_BY_CELL: dict[str, int] = {"group": 16}


def multi_start_n_starts(dataset_class: str | None = None) -> int:
    """Resolve ``n_starts`` for a cell, honouring ``SEARCHES_N_STARTS``."""
    override = os.environ.get("SEARCHES_N_STARTS")
    if override:
        return int(override)
    return _MULTI_START_N_STARTS_BY_CELL.get(dataset_class, _MULTI_START_N_STARTS)


# The JAX / optax multi-start gradient MAP optimizers, keyed by profiling
# sampler name -> the ``af`` search class. Every one runs ``n_starts`` broad
# starts in parallel (its own ``jax.vmap``) and returns the best-basin point;
# all are JAX-native and require ``use_jax=True`` (a pure-NumPy config raises).
_MULTI_START_CLASSES: dict[str, type] = {
    "multi_start_adam": af.MultiStartAdam,
    "multi_start_prodigy": af.MultiStartProdigy,
    "multi_start_prodigy_autoconv": af.MultiStartProdigy,
    "multi_start_lion": af.MultiStartLion,
    "multi_start_adabelief": af.MultiStartADABelief,
}

# Samplers that wrap their optimizer in ``af.MultiStartGradientConvergence`` so
# each start early-stops when its figure-of-merit plateaus (vs the fixed
# ``n_steps`` baseline). Prodigy is the recently-shipped auto-convergence cell.
_MULTI_START_AUTOCONV: frozenset[str] = frozenset({"multi_start_prodigy_autoconv"})

# Prodigy self-tunes its learning rate, so it takes ``learning_rate=None``; the
# fixed-rate optimizers (Adam / Lion / ADABelief) take an explicit rate.
_PRODIGY_SAMPLERS: frozenset[str] = frozenset(
    {"multi_start_prodigy", "multi_start_prodigy_autoconv"}
)


def _convergence() -> af.MultiStartGradientConvergence:
    """The auto-convergence early-stop criterion for the ``*_autoconv`` cells."""
    return af.MultiStartGradientConvergence(
        check_for_convergence=True, window=50, rtol=1e-4, atol=1e-3, min_steps=100
    )


def multi_start_settings(
    sampler: str = "multi_start_adam", dataset_class: str | None = None
) -> dict:
    """The ``n_starts`` / ``n_steps`` / ``learning_rate`` knobs a MultiStart
    builder constructs the search with.

    Exposed so ``_sampler_config_dict`` records exactly what was run. Prodigy
    variants omit ``learning_rate`` (they self-tune it). ``n_starts`` is
    per-cell (see ``multi_start_n_starts``).
    """
    settings = {
        "n_starts": multi_start_n_starts(dataset_class),
        "n_steps": _MULTI_START_N_STEPS,
    }
    if sampler not in _PRODIGY_SAMPLERS:
        settings["learning_rate"] = _MULTI_START_LEARNING_RATE
    return settings


def build_multi_start(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    config_name: str,
    use_jax: bool,
) -> af.NonLinearSearch:
    """Construct a first-class MultiStart gradient MAP search for one cell.

    Dispatches on ``sampler`` to the right ``af.MultiStart*`` class, attaching an
    ``af.MultiStartGradientConvergence`` for the ``*_autoconv`` variants (early
    stop) and leaving ``convergence=None`` (run the full ``n_steps``) otherwise.

    Unlike ``af.Nautilus`` these have no ``n_live`` and do not use the
    ``use_jax_vmap`` / ``force_x1_cpu`` ``Fitness`` path — they build their own
    batched ``value_and_grad``. ``number_of_cores=1`` is profile-convention
    metadata (the search runs a single-process vmap loop).
    """
    cls = _MULTI_START_CLASSES[sampler]
    kwargs: dict = dict(
        name=config_name,
        path_prefix=f"searches/{sampler}/{dataset_class}/{model_type}/{instrument}",
        number_of_cores=1,
        **multi_start_settings(sampler, dataset_class),
    )
    if sampler in _MULTI_START_AUTOCONV:
        kwargs["convergence"] = _convergence()
    return cls(**kwargs)


SamplerBuilder = Callable[..., af.NonLinearSearch]
SAMPLER_BUILDERS: dict[str, SamplerBuilder] = {
    "nautilus": build_nautilus,
    **{name: build_multi_start for name in _MULTI_START_CLASSES},
}
