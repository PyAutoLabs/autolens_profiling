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

import sys
from pathlib import Path
from typing import Callable

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
    ("imaging", "pixelization"): 150,
    ("imaging", "delaunay"): 150,
    ("interferometer", "mge"): 200,
    ("interferometer", "pixelization"): 150,
    ("interferometer", "delaunay"): 150,
    ("point_source", "image_plane"): 200,
    ("point_source", "source_plane"): 200,
    ("datacube", "delaunay"): 150,
}


# NSS-specific defaults. ``af.NSS`` exposes three knobs above ``n_live``:
# - ``num_mcmc_steps``: inner slice-MCMC steps per dead-point batch.
# - ``num_delete``: particles killed per outer iteration. Larger reduces
#   JIT overhead per iteration at the cost of slightly worse posterior
#   coverage.
# - ``termination``: stop when ``logZ_live - logZ < termination``. Default
#   ``-3.0`` corresponds to delta-logZ < 1e-3 — matched to PyAutoFit's
#   own production-tested default.
_NSS_DEFAULTS: dict[str, int | float] = {
    "num_mcmc_steps": 5,
    "num_delete": 50,
    "termination": -3.0,
    "seed": 42,
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


def build_nss(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    config_name: str,
    use_jax: bool,
) -> "af.NSS":
    """Construct a first-class ``af.NSS`` (JAX nested slice sampler) for one cell.

    ``af.NSS`` is JAX-native: the likelihood + prior closures both run inside
    ``jax.jit`` and execute on whatever device JAX is configured for. There
    is no NumPy / multiprocessing fallback, so:

    - ``use_jax`` must be True (the analysis must be JAX-traceable).
    - ``number_of_cores`` is accepted by ``af.NSS`` for API parity but
      ignored. We leave it at the default ``1``.
    - There is no ``force_x1_cpu`` / ``use_jax_vmap`` toggle; NSS always
      batches across live particles natively.

    ``n_live`` matches the per-(ds, model) SLaM choice used for Nautilus
    so timing comparisons line up. ``num_delete=50``, ``num_mcmc_steps=5``,
    ``termination=-3.0`` are PyAutoFit's production-tested defaults.
    """
    if not use_jax:
        raise ValueError(
            "af.NSS is JAX-native; running with use_jax=False is not "
            "supported. Unset PYAUTO_DISABLE_JAX or use sampler='nautilus' "
            "for the NumPy-front profile."
        )
    n_live = n_live_for(dataset_class, model_type)
    # NSS's ``num_delete`` plays the same role as Nautilus ``n_batch``: it
    # controls how many likelihoods fire in parallel per outer iteration.
    # Cap it at the per-cell vmap budget so heavy cells (delaunay, inversion-
    # based) don't OOM the A100. Floor at the default so small cells still
    # benefit from sane batching.
    num_delete = min(
        int(_NSS_DEFAULTS["num_delete"]),
        vmap_batch_for_cell(dataset_class, model_type, instrument),
    )
    return af.NSS(
        name=config_name,
        path_prefix=f"searches/{sampler}/{dataset_class}/{model_type}/{instrument}",
        n_live=n_live,
        num_mcmc_steps=int(_NSS_DEFAULTS["num_mcmc_steps"]),
        num_delete=num_delete,
        termination=float(_NSS_DEFAULTS["termination"]),
        seed=int(_NSS_DEFAULTS["seed"]),
    )


SamplerBuilder = Callable[..., af.NonLinearSearch]
SAMPLER_BUILDERS: dict[str, SamplerBuilder] = {
    "nautilus": build_nautilus,
    "nss": build_nss,
}
