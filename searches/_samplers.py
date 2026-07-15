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
    # Sersic is parametric like MGE, so it takes the ``source_lp[1]`` value.
    ("imaging", "sersic"): 200,
    ("imaging", "mge"): 200,
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


# Multi-start fiducial settings. Single-sourced here so the builder and the JSON
# config block (``_runner._sampler_config_dict``) record identical values.
# ``n_starts=64`` is a representative multi-start batch for a local/A100 profile
# (the GIGA-Lens recipe uses hundreds).
_MULTI_START_DEFAULTS: dict = {
    "n_starts": 64,
    "n_steps": 300,
    "learning_rate": 0.01,
}

# These three — and only these — control convergence, so they are the tuning
# axis (#69). ``batch_size`` is deliberately NOT here: it is numerically inert
# (verified on an A100 across {None,1,4,14,100}), bounds VRAM only, and is
# resolved per-cell from the vram table.
_TUNABLE = {
    "n_starts": int,
    "n_steps": int,
    "learning_rate": float,
}


def multi_start_settings() -> dict:
    """The knobs ``build_multi_start_adam`` constructs the search with.

    Exposed so ``_sampler_config_dict`` records exactly what was run.

    The optimizer-tuning campaign (#69) overrides them per run via the
    ``MULTI_START_SETTINGS`` env var, e.g.::

        MULTI_START_SETTINGS="n_starts=32,n_steps=1000,learning_rate=0.001"

    Each grid point is then its own job writing its own
    ``<sampler>/<dataset>/<model>/<instrument>/<config_name>.json`` artifact, so
    the existing sweep/aggregate machinery carries the grid without needing a
    new sweep dimension. Unset = the fiducial defaults (no behaviour change).
    """
    settings = dict(_MULTI_START_DEFAULTS)

    raw = os.environ.get("MULTI_START_SETTINGS", "").strip()
    if not raw:
        return settings

    for item in raw.split(","):
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or key not in _TUNABLE:
            raise ValueError(
                f"MULTI_START_SETTINGS: unknown or malformed entry {item!r}. "
                f"Tunable keys: {sorted(_TUNABLE)} (batch_size is not tunable — "
                "it is numerically inert and comes from the vram table)."
            )
        settings[key] = _TUNABLE[key](value)

    return settings


def build_multi_start_adam(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    config_name: str,
    use_jax: bool,
) -> af.MultiStartAdam:
    """Construct a first-class ``af.MultiStartAdam`` search for one profiling cell.

    ``MultiStartAdam`` is a JAX / ``optax`` multi-start first-order gradient MAP
    optimizer: it runs ``n_starts`` broad starts in parallel (its own ``jax.vmap``)
    and returns the best-basin point. Unlike ``af.Nautilus`` it:

    - is JAX-native and **requires** a JAX-traceable analysis (``use_jax=True``);
      a pure-NumPy config will raise. The sweep runs JAX-on by default.
    - has no ``n_live`` (it uses ``n_starts`` / ``n_steps``), and
    - does not use the ``use_jax_vmap`` / ``force_x1_cpu`` ``Fitness`` path — it
      builds its own batched ``value_and_grad``.

    ``batch_size`` comes from the same per-cell vram table Nautilus's ``n_batch``
    does (PyAutoFit#1374 added the knob): it tiles the vmapped ``value_and_grad``
    via ``jax.lax.map`` so a memory-heavy cell does not OOM. It is numerically
    inert — it bounds VRAM and nothing else — so it is set here rather than
    tuned. Without it, a pixelized cell's jvp fusion reaches 58 GiB in float64
    and exhausts an 80 GB A100.

    ``number_of_cores=1`` is kept for consistency with the profile convention
    (it is metadata here; the search runs a single-process vmap loop).
    """
    return af.MultiStartAdam(
        name=config_name,
        path_prefix=f"searches/{sampler}/{dataset_class}/{model_type}/{instrument}",
        number_of_cores=1,
        batch_size=vmap_batch_for_cell(dataset_class, model_type, instrument),
        **multi_start_settings(),
    )


SamplerBuilder = Callable[..., af.NonLinearSearch]
SAMPLER_BUILDERS: dict[str, SamplerBuilder] = {
    "nautilus": build_nautilus,
    "multi_start_adam": build_multi_start_adam,
}
