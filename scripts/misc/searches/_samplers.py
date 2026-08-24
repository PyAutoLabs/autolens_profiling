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
from searches._setup import positions_arm_tag, positions_settings  # noqa: E402
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
    # #678 phase B evidence campaign — same parametric-phase n_live as the
    # image_plane / source_plane anchors above (same lens-mass free params).
    ("point_source", "image_plane_solved"): 200,
    ("point_source", "source_plane_solved"): 200,
    ("point_source", "source_plane_tensor"): 200,
    ("point_source", "image_plane_repeat_solved"): 200,
    ("datacube", "delaunay"): 150,
    # #678 phase B evidence campaign — cluster cells. n_live=100 matches
    # autolens_workspace/scripts/cluster/modeling.py's own Nautilus default
    # for this ~6-free-parameter model (2 main lenses + scaling tier + host
    # halo), rather than the 200 used for the point_source parametric phases.
    ("cluster", "source_plane"): 100,
    ("cluster", "source_plane_solved"): 100,
    ("cluster", "source_plane_tensor"): 100,
    ("cluster", "image_plane_solved"): 100,
}


def arm_unique_tag(*parts: str | None) -> str | None:
    """Compose non-``None`` arm-tag parts into one ``unique_tag``, else ``None``.

    Shared by every sampler builder (Nautilus, NSS, MultiStart*) so a
    positions-on arm always gets a distinct ``unique_tag`` — and therefore a
    distinct output directory and identifier (PyAutoFit's identifier hashes
    ``[search, model, unique_tag]``; the ``Analysis`` object, and therefore
    whether a positions penalty is attached, is NOT hashed — see
    ``_setup.py``'s positions-plumbing docstring). Returns ``None`` only when
    every part is ``None``, so an unseeded, positions-off cell keeps its exact
    recorded output path.
    """
    resolved = [part for part in parts if part]
    return "_".join(resolved) if resolved else None


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


def nautilus_seed() -> int | None:
    """Resolve ``SEARCHES_NAUTILUS_SEED`` (default ``None`` = af.Nautilus default).

    Phase 4 Stage 2 (W2, #160) needs >= 5 Nautilus seeds per positions arm.
    ``seed`` IS an ``af.Nautilus.__identifier_fields__`` entry, so seeded arms
    get distinct autofit output directories; the submit must still carry the
    seed in ``--config-name`` so the results JSON basenames stay disjoint.
    """
    raw = os.environ.get("SEARCHES_NAUTILUS_SEED")
    return int(raw) if raw else None


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
    seed = nautilus_seed()
    return af.Nautilus(
        name=config_name,
        seed=seed,
        path_prefix=f"searches/{sampler}/{dataset_class}/{model_type}/{instrument}",
        n_live=n_live,
        n_batch=n_batch,
        number_of_cores=1,
        force_x1_cpu=use_jax,
        use_jax_vmap=use_jax,
        force_pickle_overwrite=True,
        iterations_per_update=3 * n_live,
        # See arm_unique_tag's docstring: a positions-on arm MUST carry its own
        # tag or it silently shares an output directory / identifier with the
        # positions-off cell (the Analysis object is not hashed).
        unique_tag=arm_unique_tag(positions_arm_tag()),
    )


# NSS (nested slice sampling, ``af.NSS`` on mainline ``blackjax.nss``) profiling
# settings. Single-sourced here so the builder and the JSON config block record
# identical values, mirroring the MultiStart pattern below.
#
# The defaults REPLICATE THE FORK-ERA HISTORY (num_mcmc_steps=5, num_delete=50,
# termination=-3.0, seed=42) so the first mainline rows are directly comparable
# to the recorded fork rows (results/searches/nss/, v2026.5.21.1). The Phase 2
# campaign (PROGRAMME.md §4) scans away from these via the env overrides —
# in particular H2.1 pre-registers num_mcmc_steps >= max(5, 2*dim) as the fix
# for the +7-13 nat logZ bias, so scan arms set SEARCHES_NSS_NUM_MCMC_STEPS.
_NSS_NUM_MCMC_STEPS = 5
_NSS_NUM_DELETE = 50
_NSS_TERMINATION = -3.0
_NSS_SEED = 42


def nss_settings() -> dict:
    """The ``af.NSS`` knobs a profiling cell constructs the search with.

    Exposed so ``_sampler_config_dict`` records exactly what was run. Every
    knob honours a ``SEARCHES_NSS_*`` env override so Phase 2 scan arms
    (n_live x num_delete x inner steps x dlogz) drive the same builder.
    ``chunk_size`` stays ``None`` unless overridden — the GPU-memory chunking
    path is bit-identical to unchunked at fixed seed (PyAutoFit PR#1492) but
    is a separate lever from the science knobs.
    """
    chunk = os.environ.get("SEARCHES_NSS_CHUNK_SIZE")
    return {
        "num_mcmc_steps": int(os.environ.get("SEARCHES_NSS_NUM_MCMC_STEPS", _NSS_NUM_MCMC_STEPS)),
        "num_delete": int(os.environ.get("SEARCHES_NSS_NUM_DELETE", _NSS_NUM_DELETE)),
        "chunk_size": int(chunk) if chunk else None,
        "termination": float(os.environ.get("SEARCHES_NSS_TERMINATION", _NSS_TERMINATION)),
        "seed": int(os.environ.get("SEARCHES_NSS_SEED", _NSS_SEED)),
    }


def build_nss(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    config_name: str,
    use_jax: bool,
) -> af.NonLinearSearch:
    """Construct a first-class ``af.NSS`` search for one profiling cell.

    ``af.NSS`` is JAX-native (the whole sampler loop runs inside ``jax.jit``),
    so a pure-NumPy config is a contradiction and raises rather than silently
    profiling nothing. ``n_live`` comes from the same SLaM-mirror table as
    Nautilus so the two nested samplers profile the same target at the same
    live-point budget; ``SEARCHES_NSS_N_LIVE`` overrides for the Phase 2
    n_live scan.
    """
    if not use_jax:
        raise ValueError(
            "af.NSS is JAX-native; a PYAUTO_DISABLE_JAX=1 profiling config cannot run it."
        )
    n_live = int(os.environ.get("SEARCHES_NSS_N_LIVE", n_live_for(dataset_class, model_type)))
    return af.NSS(
        name=config_name,
        path_prefix=f"searches/{sampler}/{dataset_class}/{model_type}/{instrument}",
        n_live=n_live,
        number_of_cores=1,
        # See arm_unique_tag's docstring / build_nautilus's comment above.
        unique_tag=arm_unique_tag(positions_arm_tag()),
        **nss_settings(),
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
# Cell keys are ``dataset_class`` or the more specific
# ``dataset_class:model_type`` (resolved first). The pixelized cells run 16
# starts: their per-replica inversion is the memory driver (the 58 GB jvp
# citation below) and the #117 campaign showed 16 broad starts suffice to
# recover the truth basin on every searchable mesh.
_MULTI_START_N_STARTS_BY_CELL: dict[str, int] = {
    "group": 32,
    "imaging:pixelization": 16,
    "imaging:knn": 16,
    "imaging:delaunay_matern": 16,
}

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
# For the PIXELIZED cells batching is NOT optional: the unbatched 16-start
# jvp fusion is the ~58 GB allocation cited above, so ``batch_size=4`` (the
# #117 campaign value, numerically inert per PyAutoFit#1374) is the default.
_MULTI_START_BATCH_BY_CELL: dict[str, int] = {
    "imaging:pixelization": 4,
    "imaging:knn": 4,
    "imaging:delaunay_matern": 4,
}


def _cell_lookup(table: dict[str, int], dataset_class: str | None, model_type: str | None):
    """Resolve a per-cell knob: ``dataset_class:model_type`` wins over
    ``dataset_class`` wins over the module default (returned as ``None``)."""
    if dataset_class and model_type and f"{dataset_class}:{model_type}" in table:
        return table[f"{dataset_class}:{model_type}"]
    return table.get(dataset_class)


def multi_start_n_starts(dataset_class: str | None = None, model_type: str | None = None) -> int:
    """Resolve ``n_starts`` for a cell, honouring ``SEARCHES_N_STARTS``."""
    override = os.environ.get("SEARCHES_N_STARTS")
    if override:
        return int(override)
    v = _cell_lookup(_MULTI_START_N_STARTS_BY_CELL, dataset_class, model_type)
    return v if v is not None else _MULTI_START_N_STARTS


# Per-dataset-class ``n_steps``. The 300-step default is far too few for the
# group cell: a 32-start adam run stopped on ``max_steps`` with
# ``converged: false`` while its figure-of-merit was still falling 7.2% over the
# final 50 steps (747335 -> 464003, still descending). Any "gradient optimizers
# can't do this model" claim read off a 300-step run would be an artefact of the
# step budget, not a property of the method. ``SEARCHES_N_STEPS`` overrides.
# The pixelized cells get the same 3000-step budget for a different reason
# (#117): with a FREE regularization the best-fit reg mode is found by
# resurrection crossing modes, which landed at step ~1300 (knn) / ~2000
# (delaunay+AdaptSplit) — a long plateau is a reg mode, not convergence. A
# 300-step read of these cells would be an artefact of the budget.
_MULTI_START_N_STEPS_BY_CELL: dict[str, int] = {
    "group": 3000,
    "imaging:pixelization": 3000,
    "imaging:knn": 3000,
    "imaging:delaunay_matern": 3000,
}


def multi_start_n_steps(dataset_class: str | None = None, model_type: str | None = None) -> int:
    """Resolve ``n_steps`` for a cell, honouring ``SEARCHES_N_STEPS``."""
    override = os.environ.get("SEARCHES_N_STEPS")
    if override:
        return int(override)
    v = _cell_lookup(_MULTI_START_N_STEPS_BY_CELL, dataset_class, model_type)
    return v if v is not None else _MULTI_START_N_STEPS


# Prior-support enforcement arms (PyAutoFit#1477). ``none`` is the library
# default and is bit-identical to a search built with no ``clipper`` at all;
# ``prior_box`` projects each step back onto the prior support.
#
# Recorded as a STRING in the sampler config, not as the object: the config dict
# is serialised straight into the results JSON, and a Clipper instance is not
# JSON-serialisable. ``build_multi_start`` maps the label to the object.
_MULTI_START_CLIPPERS: dict[str, str] = {
    "none": "ClipperNone",
    "prior_box": "ClipperPriorBox",
}


def multi_start_clipper() -> str:
    """Resolve the clipper arm label, honouring ``SEARCHES_CLIPPER``.

    Defaults to ``none`` so every pre-existing cell keeps its recorded numbers.
    """
    label = os.environ.get("SEARCHES_CLIPPER", "none").strip().lower()
    if label not in _MULTI_START_CLIPPERS:
        raise ValueError(
            f"SEARCHES_CLIPPER={label!r} is not one of {sorted(_MULTI_START_CLIPPERS)}"
        )
    return label


def _clipper_object(label: str):
    """The ``af`` clipper instance for a resolved arm label."""
    if label == "none":
        return af.ClipperNone()
    return af.ClipperPriorBox()


# Per-parameter step-scaling arms (PyAutoFit#1483). ``none`` is the library
# default and is bit-identical to a search built with no ``scaler`` at all;
# ``prior_width`` derives a diagonal preconditioner from the priors.
#
# Recorded as a STRING for the same reason as the clipper: the config dict is
# serialised straight into the results JSON and a Scaler instance is not
# JSON-serialisable.
_MULTI_START_SCALERS: dict[str, str] = {
    "none": "ScalerNone",
    "prior_width": "ScalerPriorWidth",
}


def multi_start_scaler() -> str:
    """Resolve the scaler arm label, honouring ``SEARCHES_SCALER``.

    Defaults to ``none`` so every pre-existing cell keeps its recorded numbers.
    """
    label = os.environ.get("SEARCHES_SCALER", "none").strip().lower()
    if label not in _MULTI_START_SCALERS:
        raise ValueError(f"SEARCHES_SCALER={label!r} is not one of {sorted(_MULTI_START_SCALERS)}")
    return label


def _scaler_object(label: str):
    """The ``af`` scaler instance for a resolved arm label."""
    if label == "none":
        return af.ScalerNone()
    return af.ScalerPriorWidth()


def multi_start_seed() -> int | None:
    """Resolve the search's own RNG seed, honouring ``SEARCHES_SEED``.

    ``None`` (the default) keeps the library's historical fixed draw, so every
    pre-existing cell reproduces its recorded numbers. The seed reaches the
    broad-start draw and the resurrection redraw — the two places a multi-seed
    reliability study actually differs — so a repeated-seed campaign
    (``PROGRAMME.md`` §3: "reliability is P(correct | fixed budget), measured
    over >= 5 seeds") MUST set it. Before the search had a ``seed`` the start
    band was a hardcoded ``default_rng(0)`` and a multi-seed study was silently
    single-seed.
    """
    raw = os.environ.get("SEARCHES_SEED")
    return int(raw) if raw not in (None, "") else None


def multi_start_unique_tag(
    dataset_class: str | None = None, model_type: str | None = None
) -> str | None:
    """A per-arm ``unique_tag`` for seeded multi-start runs, else ``None``.

    **This is a correctness guard, not cosmetics.**
    ``AbstractMultiStartGradient.__identifier_fields__`` is ``("clipper",)``,
    and when a class declares identifier fields the ``Identifier`` hash uses
    ONLY those. So ``n_starts``, ``n_steps``, ``seed``, ``scaler`` and the
    convergence criterion do **not** enter the search identifier: two arms of a
    reliability scan differing only in seed resolve to the same output
    directory, and the ``.completed`` short-circuit then makes ``fit()`` return
    the first arm's cached result without ever entering ``_fit``. The scan
    would report one run's numbers five times.

    ``unique_tag`` is appended to ``identifier_list`` in
    ``AbstractPaths._identifier`` *and* sits in ``output_path`` above ``name``,
    so tagging fixes both the hash and the directory.

    Also composes in ``_setup.positions_arm_tag()`` (Phase 4 Stage 1, issue
    #159) for the SAME reason: the ``Analysis`` object — and therefore whether
    a ``positions_likelihood_list`` is attached — is not part of the
    identifier hash either. Returned whenever a seed is set OR positions are
    on, so an unseeded-but-positions-on cell still gets a tag (never silently
    shares a positions-off cell's output path); ``None`` only when neither
    applies, so the ordinary unseeded/positions-off cell keeps byte-identical
    output paths to its recorded runs.
    """
    seed = multi_start_seed()
    pos_tag = positions_arm_tag()
    if seed is None and pos_tag is None:
        return None
    seed_tag = (
        f"n{multi_start_n_starts(dataset_class, model_type)}"
        f"_s{multi_start_n_steps(dataset_class, model_type)}"
        f"_seed{seed}"
        if seed is not None
        else None
    )
    return arm_unique_tag(seed_tag, pos_tag)


def multi_start_batch_size(
    dataset_class: str | None = None, model_type: str | None = None
) -> int | None:
    """Resolve the memory-bounding ``batch_size``, honouring ``SEARCHES_BATCH_SIZE``.

    ``None`` (the default for every cell but ``group``) keeps the fastest
    unbatched single-vmap path.
    """
    override = os.environ.get("SEARCHES_BATCH_SIZE")
    if override:
        return int(override) or None
    return _cell_lookup(_MULTI_START_BATCH_BY_CELL, dataset_class, model_type)


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
    sampler: str = "multi_start_adam",
    dataset_class: str | None = None,
    model_type: str | None = None,
) -> dict:
    """The ``n_starts`` / ``n_steps`` / ``learning_rate`` knobs a MultiStart
    builder constructs the search with.

    Exposed so ``_sampler_config_dict`` records exactly what was run. Prodigy
    variants omit ``learning_rate`` (they self-tune it). ``n_starts`` is
    per-cell (see ``multi_start_n_starts``).
    """
    settings: dict = {
        "n_starts": multi_start_n_starts(dataset_class, model_type),
        "n_steps": multi_start_n_steps(dataset_class, model_type),
    }
    batch_size = multi_start_batch_size(dataset_class, model_type)
    if batch_size is not None:
        settings["batch_size"] = batch_size
    if sampler not in _PRODIGY_SAMPLERS:
        settings["learning_rate"] = _MULTI_START_LEARNING_RATE
    # A string label, so the recorded config stays JSON-serialisable; the arm
    # is always recorded, including the ``none`` default, so a result file can
    # never be ambiguous about which arm produced it.
    settings["clipper"] = multi_start_clipper()
    settings["scaler"] = multi_start_scaler()
    # Always recorded, including the ``None`` default: a seeded and an unseeded
    # run must be distinguishable in the artifact, not both read as "no key".
    settings["seed"] = multi_start_seed()
    # Always recorded, including {"enabled": False}: a positions-on and
    # positions-off run must be distinguishable in the artifact (Phase 4
    # Stage 1, issue #159).
    settings["positions"] = positions_settings()
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
    settings = multi_start_settings(sampler, dataset_class, model_type)
    # Swap the recorded string label for the live object. Done here rather than
    # in ``multi_start_settings`` so that function stays serialisable — it is
    # what ``_sampler_config_dict`` writes into the results JSON.
    settings["clipper"] = _clipper_object(settings["clipper"])
    settings["scaler"] = _scaler_object(settings["scaler"])
    # "positions" is a recorded-config-only block (consumed by
    # _sampler_config_dict / multi_start_settings' JSON shape); no
    # af.MultiStart* class accepts it as a constructor kwarg.
    settings.pop("positions", None)
    kwargs: dict = dict(
        name=config_name,
        path_prefix=f"searches/{sampler}/{dataset_class}/{model_type}/{instrument}",
        # Only ``clipper`` enters this search's identifier, so a seeded arm has
        # to carry its own tag or a sibling arm's completed fit short-circuits
        # it — see ``multi_start_unique_tag``. ``None`` for unseeded cells,
        # which keeps their output paths exactly as recorded.
        unique_tag=multi_start_unique_tag(dataset_class, model_type),
        number_of_cores=1,
        convergence=_convergence(autoconv=sampler in _MULTI_START_AUTOCONV),
        **settings,
    )
    return cls(**kwargs)


SamplerBuilder = Callable[..., af.NonLinearSearch]
SAMPLER_BUILDERS: dict[str, SamplerBuilder] = {
    "nautilus": build_nautilus,
    "nss": build_nss,
    **{name: build_multi_start for name in _MULTI_START_CLASSES},
}


def assert_disjoint_output_paths(
    search_a: af.NonLinearSearch, search_b: af.NonLinearSearch
) -> None:
    """Assert two constructed searches have BOTH a different ``output_path``
    and a different ``identifier``.

    A standalone correctness guard for every ``unique_tag`` composition above
    (``arm_unique_tag`` / ``multi_start_unique_tag`` / the ``build_nautilus`` &
    ``build_nss`` positions wiring). PyAutoFit's identifier hashes only
    ``[search, model, unique_tag]`` — the ``Analysis`` object is never part of
    the hash, so two arms that differ only in what's attached to the analysis
    (e.g. a ``positions_likelihood_list``) MUST differ in ``unique_tag`` or
    they silently share one output directory / identifier, and the second
    arm's ``search.fit()`` short-circuits to the first arm's cached
    ``.completed`` result instead of actually running.
    """
    if search_a.paths.output_path == search_b.paths.output_path:
        raise AssertionError(
            f"output_path collision between two arms expected to be distinct: "
            f"{search_a.paths.output_path!r}"
        )
    if search_a.paths.identifier == search_b.paths.identifier:
        raise AssertionError(
            f"identifier collision between two arms expected to be distinct: "
            f"{search_a.paths.identifier!r}"
        )
