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


import os
import sys
from collections.abc import Callable
from pathlib import Path

import autofit as af

# ``vram/config.py`` lives at the workspace root and stores per-(dataset, model,
# instrument) A100-probed vmap batch sizes. The samplers read it so we don't
# hardcode batch sizes that would OOM on heavier cells (e.g. imaging/delaunay
# at HST scale uses batch=16, not the Nautilus-default 100).
_WORKSPACE_ROOT = _profiling_root()
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

# Per-dataset-class ``n_starts``. For a multi-start gradient search the starts
# ARE the natural batch (particle) dimension, so a plain ``jax.vmap`` over them
# is both the fastest and the structurally simplest path. The group cell's
# replica is large (4 lenses + 4 sources = 54 params, 8 MGE bases through one
# inversion), so on a 6 GB laptop GPU we size the number of starts to fit rather
# than chunking them (see the batch_size note below). An A100 runs the full 64:
#   SEARCHES_N_STARTS=64 python searches/multi_start_adam/group/mge.py ...
_MULTI_START_N_STARTS_BY_CELL: dict[str, int] = {"group": 32}

# ``batch_size`` (jax.lax.map chunking) is deliberately NOT used for the group
# cell. It is a genuine memory lever in MultiStartGradient — aimed at
# likelihoods whose batched jvp cannot fit at all (its docstring cites a
# pixelized source at 16 starts, ~58 GB) — and it is numerically identical to
# the vmap. But measured on this cell it is the wrong trade: the scan it adds
# across chunks costs a lot of compile time for no scientific gain.
#   16 starts, unbatched vmap : 13 min 35 s to compile
#   64 starts + batch_size=8  : >44 min, still compiling
# So we take the smaller vmap instead. ``SEARCHES_BATCH_SIZE`` still forces it
# on for a cell that genuinely cannot fit any workable n_starts.
_MULTI_START_BATCH_BY_CELL: dict[str, int] = {}


def multi_start_n_starts(dataset_class: str | None = None) -> int:
    """Resolve ``n_starts`` for a cell, honouring ``SEARCHES_N_STARTS``."""
    override = os.environ.get("SEARCHES_N_STARTS")
    if override:
        return int(override)
    return _MULTI_START_N_STARTS_BY_CELL.get(dataset_class, _MULTI_START_N_STARTS)


# Per-dataset-class ``n_steps``. The 300-step default is far too few for the
# group cell: a 32-start adam run stopped on ``max_steps`` with
# ``converged: false`` while its figure-of-merit was still falling 7.2% over the
# final 50 steps (747335 -> 464003, still descending). Any "gradient optimizers
# can't do this model" claim read off a 300-step run would be an artefact of the
# step budget, not a property of the method. ``SEARCHES_N_STEPS`` overrides.
_MULTI_START_N_STEPS_BY_CELL: dict[str, int] = {"group": 3000}


def multi_start_n_steps(dataset_class: str | None = None) -> int:
    """Resolve ``n_steps`` for a cell, honouring ``SEARCHES_N_STEPS``."""
    override = os.environ.get("SEARCHES_N_STEPS")
    if override:
        return int(override)
    return _MULTI_START_N_STEPS_BY_CELL.get(dataset_class, _MULTI_START_N_STEPS)


def multi_start_batch_size(dataset_class: str | None = None) -> int | None:
    """Resolve the memory-bounding ``batch_size``, honouring ``SEARCHES_BATCH_SIZE``.

    ``None`` (the default for every cell but ``group``) keeps the fastest
    unbatched single-vmap path.
    """
    override = os.environ.get("SEARCHES_BATCH_SIZE")
    if override:
        return int(override) or None
    return _MULTI_START_BATCH_BY_CELL.get(dataset_class)


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

# Samplers that run with auto-convergence early-stopping ON (vs a genuine
# fixed-``n_steps`` baseline). Prodigy is the recently-shipped auto-convergence
# cell.
#
# IMPORTANT: ``convergence=None`` does NOT mean "no convergence checking" — the
# search defaults it ON (samples_info.json from a plain adam run records
# check_for_convergence: true, window 50, rtol 1e-4, atol 1e-3, min_steps 100).
# So a genuine fixed-step arm must explicitly pass check_for_convergence=False;
# otherwise the "fixed" and "autoconv" cells are the *same run* and the A/B is
# vacuous.
_MULTI_START_AUTOCONV: frozenset[str] = frozenset({"multi_start_prodigy_autoconv"})

# Prodigy self-tunes its learning rate, so it takes ``learning_rate=None``; the
# fixed-rate optimizers (Adam / Lion / ADABelief) take an explicit rate.
_PRODIGY_SAMPLERS: frozenset[str] = frozenset(
    {"multi_start_prodigy", "multi_start_prodigy_autoconv"}
)


def _convergence(autoconv: bool) -> af.MultiStartGradientConvergence:
    """The convergence criterion for a MultiStart cell.

    ``autoconv=True`` → early-stop when each start's figure-of-merit plateaus
    (these are the search's own defaults, passed explicitly so the recorded
    config is self-describing).

    ``autoconv=False`` → **genuinely** fixed-step: ``check_for_convergence`` is
    switched OFF so the run always completes ``n_steps``. This must be passed
    explicitly — leaving ``convergence=None`` silently enables checking, which
    would make the fixed-step and autoconv cells the same run.
    """
    if autoconv:
        return af.MultiStartGradientConvergence(
            check_for_convergence=True, window=50, rtol=1e-4, atol=1e-3, min_steps=100
        )
    return af.MultiStartGradientConvergence(check_for_convergence=False)


def multi_start_settings(
    sampler: str = "multi_start_adam", dataset_class: str | None = None
) -> dict:
    """The ``n_starts`` / ``n_steps`` / ``learning_rate`` knobs a MultiStart
    builder constructs the search with.

    Exposed so ``_sampler_config_dict`` records exactly what was run. Prodigy
    variants omit ``learning_rate`` (they self-tune it). ``n_starts`` is
    per-cell (see ``multi_start_n_starts``).
    """
    settings: dict = {
        "n_starts": multi_start_n_starts(dataset_class),
        "n_steps": multi_start_n_steps(dataset_class),
    }
    batch_size = multi_start_batch_size(dataset_class)
    if batch_size is not None:
        settings["batch_size"] = batch_size
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

    Dispatches on ``sampler`` to the right ``af.MultiStart*`` class. An explicit
    ``af.MultiStartGradientConvergence`` is **always** attached: early-stopping
    for the ``*_autoconv`` variants, and ``check_for_convergence=False`` for the
    fixed-step ones. Never leave it as ``None`` — that silently enables checking
    and collapses the two arms into the same run (see ``_convergence``).

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
        convergence=_convergence(autoconv=sampler in _MULTI_START_AUTOCONV),
        **multi_start_settings(sampler, dataset_class),
    )
    return cls(**kwargs)


SamplerBuilder = Callable[..., af.NonLinearSearch]
SAMPLER_BUILDERS: dict[str, SamplerBuilder] = {
    "nautilus": build_nautilus,
    **{name: build_multi_start for name in _MULTI_START_CLASSES},
}
