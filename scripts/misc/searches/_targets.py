"""Targets registry + schema-v2 identity hashing (W4 / issue #161, Phase 1).

A **target** is (dataset_class, model_type, instrument, positions arm,
precision) — the thing a search is run against, independent of which
algorithm/config/seed does the running. ``TARGETS`` is the single canonical
registry: every profiling cell that wants to be comparable across years
should be built via ``searches._setup.build_for_cell(target=TARGETS[key])``
rather than hand-assembling the same fields, so its identity is provably
reproducible via :func:`target_id`.

Design (PROGRAMME.md §"Phase 1 — Standard benchmark matrix & targets
registry", §5 "Benchmark & result schema (v2)"):

- :class:`Target` is a frozen, hashable-by-identity spec: WHAT to build
  (dataset_class/model_type/instrument/builder), WHICH arm (positions,
  precision, log_det_method), how the truth-anchor step should treat it, and
  the :class:`Tolerances` its runs are judged against.
- :func:`target_id` hashes everything that makes two runs comparable — model
  shape, priors, likelihood settings, positions config, the dataset files on
  disk, the mask/over-sampling recipe, and (for pixelized targets) the cached
  adapt image — into a short, stable, cross-process identifier. Two runs
  with the same ``target_id`` are provably the same target; two with
  different ids differ in a way this function can point to.
- :func:`target_block` is what a results JSON's ``target`` key (schema v2)
  gets set to.

**Prior repr is NOT deterministic** — ``repr(af.UniformPrior(...))`` embeds
the process-local ``Prior.id`` counter (``<UniformPrior id=42>``), which
differs across processes and even across two calls in the same process. The
original ``slogdet_ab.py`` prototype this module lifts ``_hash``/
``target_block`` from used ``repr(prior)`` and would silently have baked a
non-reproducible id into every ``target_id``. This module instead
canonicalises each prior via ``prior.dict()`` with the ``"id"`` key
stripped — content-only (class name + numeric fields), verified identical
across two freshly-constructed priors with the same parameters.
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


import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import autolens as al  # noqa: E402

_WORKSPACE_ROOT = _profiling_root()
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from searches import _setup  # noqa: E402

# -----------------------------------------------------------------------------
# Tolerances
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """Pass/fail thresholds a target's runs are judged against.

    Defaults are the PROGRAMME.md §"Phase 1" pass/fail metrics: Δ(max logL)
    vs truth bar <= 2 nats, per-parameter posterior mean shift <= 0.2σ_ref,
    σ ratio in [0.8, 1.25] vs a reference posterior. ``per_parameter_recovery``
    is empty by default (no per-parameter recovery tolerance registered) —
    a target can supply e.g. ``{"einstein_radius": 0.01}`` (absolute
    tolerance in the parameter's own units) to add one.
    """

    delta_max_ll_nats: float = 2.0
    mean_shift_sigma: float = 0.2
    sigma_ratio: tuple[float, float] = (0.8, 1.25)
    per_parameter_recovery: dict[str, float] = field(default_factory=dict)


_DEFAULT_TOLERANCES = Tolerances()


# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------

# model_type -> the searches._setup builder function name, uniformly
# "_<model_type>_model" for every registry-eligible model_type today.
_REGISTRY_MODEL_TYPES: tuple[str, ...] = (
    "mge",
    "delaunay",
    "delaunay_nn",
    "knn",
    "delaunay_matern",
    "pixelization",
    "slam_source_pix",
    "slam_source_pix_nn",
)

# Positions-on arms that use SLaM's own auto-threshold convention rather than
# the fixed default (mirrors the same set _setup.build_for_cell checks).
_SLAM_AUTO_THRESHOLD_MODEL_TYPES = frozenset({"slam_source_pix", "slam_source_pix_nn"})


def _builder_for(model_type: str) -> Callable[..., Any]:
    """Resolve a registry model_type to its ``searches._setup`` builder function."""
    attr = f"_{model_type}_model"
    try:
        return getattr(_setup, attr)
    except AttributeError as exc:
        raise KeyError(
            f"No builder {attr!r} on searches._setup for model_type={model_type!r}."
        ) from exc


@dataclass(frozen=True)
class Target:
    """One canonical (dataset_class, model_type, instrument, arm) spec.

    ``key`` is the registry key this target is stored under in ``TARGETS``
    (``<model_type>[_pos]_<precision>``, e.g. ``"mge_fp64"`` /
    ``"mge_pos_mp"``) — kept ON the object (not just as the dict key) so a
    ``Target`` pulled out of the registry is self-describing.
    """

    name: str
    dataset_class: str
    model_type: str
    instrument: str
    builder: Callable[..., Any]
    positions: str  # "off" | "on"
    precision: str  # "fp64" | "mp"
    log_det_method: str | None
    truth_anchor_kind: str  # "imaging" | "point_source" | "cluster" | "none"
    tolerances: Tolerances = _DEFAULT_TOLERANCES
    notes: str = ""

    def __post_init__(self) -> None:
        if self.positions not in ("off", "on"):
            raise ValueError(f"positions must be 'off' or 'on', got {self.positions!r}")
        if self.precision not in ("fp64", "mp"):
            raise ValueError(f"precision must be 'fp64' or 'mp', got {self.precision!r}")

    @property
    def cell(self) -> str:
        return f"{self.dataset_class}/{self.model_type}/{self.instrument}"


def _target_key(model_type: str, positions: str, precision: str) -> str:
    suffix = "_pos" if positions == "on" else ""
    return f"{model_type}{suffix}_{precision}"


# W4 / issue #161 verification (2026-08-24, CPU, use_jax off, broad prior
# draws): DelaunayNN-based targets resample far more often than the Delaunay
# baseline at the SAME draws. 8 random draws in [0.2, 0.8] unit-cube: `delaunay`
# 8/8 finite; `delaunay_nn` 3/8 finite, 3/8 NaN, 2/8 FitException;
# `slam_source_pix_nn` (DelaunayNN + free reg.Adapt) 1/8 finite, 2/8 NaN, 5/8
# FitException. This is NOT a bug in the target definition — the human call
# (DECISIONS.md 2026-08-24) is explicit that DelaunayNN is a REAL target, and
# an elevated resample rate at broad, untuned priors is itself a legitimate
# Phase-1 finding, not something this registry silently works around by
# swapping mesh/regularization. Recorded on both targets' `notes` so any
# reader of the registry sees it without re-deriving it.
_DELAUNAY_NN_RESAMPLE_NOTE = (
    "Elevated resample rate observed vs the `delaunay` baseline at broad, "
    "untuned prior draws (CPU, 2026-08-24 W4 verification) — see "
    "DECISIONS.md. Not a target-definition bug; a Phase-1 finding."
)


def _build_targets() -> dict[str, Target]:
    targets: dict[str, Target] = {}
    for model_type in _REGISTRY_MODEL_TYPES:
        for positions in ("off", "on"):
            for precision in ("fp64", "mp"):
                key = _target_key(model_type, positions, precision)
                notes = (
                    _DELAUNAY_NN_RESAMPLE_NOTE
                    if model_type in ("delaunay_nn", "slam_source_pix_nn")
                    else ""
                )
                targets[key] = Target(
                    name=key,
                    dataset_class="imaging",
                    model_type=model_type,
                    instrument="hst",
                    builder=_builder_for(model_type),
                    positions=positions,
                    precision=precision,
                    # None = packaged default (cholesky). No registry axis
                    # varies log_det_method today — a slogdet A/B constructs
                    # its own Target (or uses build_ab_for_cell) rather than
                    # reading one from TARGETS.
                    log_det_method=None,
                    truth_anchor_kind="imaging",
                    tolerances=_DEFAULT_TOLERANCES,
                    notes=notes,
                )
    return targets


TARGETS: dict[str, Target] = _build_targets()


# -----------------------------------------------------------------------------
# target_id — canonical identity hashing
# -----------------------------------------------------------------------------

# Dataset files whose content bears on a target's identity. positions.json
# may be absent for an off-positions target that has never had truth
# positions derived yet — recorded as null, not skipped, so its presence
# still participates in the hash once it exists.
_DATASET_FILES: tuple[str, ...] = (
    "data.fits",
    "noise_map.fits",
    "psf.fits",
    "tracer.json",
    "positions.json",
)

# The fixed over-sampling recipe every imaging target's dataset is built
# with (see searches._setup._build_imaging). Not a variable of Target today;
# recorded literally here rather than re-derived from a live dataset object
# so target_id can be computed from (target, model, dataset_path) alone.
_OVER_SAMPLE_RECIPE: dict[str, Any] = {
    "sub_size_list": [4, 2, 1],
    "radial_list": [0.3, 0.6],
    "centre_list": [[0.0, 0.0]],
}

_FILE_HASH_CACHE: dict[Path, str | None] = {}


def _sha256_file(path: Path) -> str | None:
    """sha256 hex digest of ``path``'s contents, or ``None`` if it doesn't exist.

    Cached per resolved path (Step 1 of the plan: "cache per path") — a
    ``target_id`` call touches up to 5 dataset files, and a sweep computes
    ``target_id`` once per (target, seed, config), so without a cache the
    same multi-MB ``.fits`` files get re-hashed on every call.
    """
    path = Path(path)
    key = path.resolve() if path.exists() else path
    if key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[key]
    if not path.exists():
        _FILE_HASH_CACHE[key] = None
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _FILE_HASH_CACHE[key] = result
    return result


def _canonical_prior(prior: Any) -> dict:
    """Content-only prior spec: ``prior.dict()`` with the process-local ``id`` stripped.

    ``prior.dict()`` (e.g. ``{"type": "Uniform", "id": 3, "lower_limit": 0.0,
    "upper_limit": 8.0}``) already carries everything needed to reconstruct
    the prior; ``id`` is a monotonically-increasing counter assigned at
    construction time and is NOT part of the prior's identity — two
    independently-constructed ``UniformPrior(0.0, 8.0)`` differ in ``id`` but
    must hash identically here.
    """
    d = dict(prior.dict())
    d.pop("id", None)
    return d


def _likelihood_block(target: Target) -> dict:
    settings = al.Settings(
        use_border_relocator=target.model_type in _setup._PIX_MODEL_TYPES,
        use_mixed_precision=target.precision == "mp",
        log_det_method=target.log_det_method,
    )
    return {
        "log_det_method": target.log_det_method,
        "positive_only": bool(settings.use_positive_only_solver),
        "border_relocator": bool(settings.use_border_relocator),
        "curvature_floor": float(settings.no_regularization_add_to_curvature_diag_value),
        "regularization_term_method": settings.regularization_term_method,
        "precision": target.precision,
    }


def _positions_block(target: Target, positions_setup: dict | None = None) -> dict:
    """The target's ``positions`` block, independent of ambient env state.

    Mirrors ``searches._setup.positions_settings()``'s shape (built via the
    same ``positions_settings_for`` this module's registration prompted —
    see ``_setup.py``). The free-text ``"note"`` key is stripped (Step 1 of
    the plan: "positions (minus free-text note)") — it is documentation, not
    part of the target's identity, and changing its wording must never
    change ``target_id``.

    ``positions_setup`` is the **resolved** positions configuration this run
    actually used — the exact dict ``_setup.positions_settings()`` returns
    and ``_runner`` records verbatim as the results JSON's top-level
    ``"positions"`` key. Pass it whenever it is known.

    Why it is a parameter and not read from the environment (2026-08-27,
    issue #182): ``Target.positions`` is only ``"off"``/``"on"`` — the arm's
    threshold mode, threshold value and factor live in
    ``SEARCHES_POSITIONS_THRESHOLD`` / ``SEARCHES_POSITIONS_FACTOR`` and are
    NOT registry axes. This function used to fall back to the module
    defaults (``fixed``/``0.3``/``1e8``) unconditionally, so the Phase-4
    diagnostic arms ``pos_t0.3_f1e8``, ``pos_t0.3_f1e5`` and
    ``pos_tauto0.2_f1e8`` — three genuinely different objectives, correctly
    given three distinct ``positions_arm_tag()``s and three distinct output
    directories — all hashed to the SAME ``target_id``. Reading the env vars
    here instead would have fixed the collision at the cost of the property
    the whole module exists for: a ``target_id`` recomputable from a recorded
    artifact in any process. Taking the resolved block as an argument keeps
    both — the caller resolves the environment ONCE (``_runner``, at run
    time), records it in the artifact, and a later reader re-derives the same
    id from that record.

    Passing ``None`` keeps the historical default-derived block, so every
    positions-off row and every positions-on row that really did run at the
    ``fixed``/``0.3``/``1e8`` defaults keeps the exact ``target_id`` it was
    recorded with.
    """
    if target.positions == "off":
        if positions_setup is not None and positions_setup.get("enabled"):
            raise ValueError(
                f"target {target.name!r} has positions='off' but positions_setup says "
                f"enabled=True — refusing to hash a positions block that contradicts "
                f"the target."
            )
        block = _setup.positions_settings_for(enabled=False)
    else:
        if positions_setup is not None and not positions_setup.get("enabled"):
            raise ValueError(
                f"target {target.name!r} has positions='on' but positions_setup says "
                f"enabled=False — refusing to hash a positions block that contradicts "
                f"the target."
            )
        if positions_setup is None:
            mode = "auto" if target.model_type in _SLAM_AUTO_THRESHOLD_MODEL_TYPES else "fixed"
            fixed_value = None if mode == "auto" else float(_setup._POSITIONS_THRESHOLD_DEFAULT)
            factor = float(_setup._POSITIONS_FACTOR_DEFAULT)
        else:
            mode = positions_setup["threshold_mode"]
            fixed_value = float(positions_setup["threshold_value"]) if mode == "fixed" else None
            factor = float(positions_setup["factor"])
        # Rebuilt through positions_settings_for rather than copied: a recorded
        # block that gained (or lost) a key must not perturb the hash, and an
        # unrecognised threshold_mode must raise rather than hash silently.
        block = _setup.positions_settings_for(
            enabled=True,
            mode=mode,
            fixed_value=fixed_value,
            factor=factor,
        )
    block = dict(block)
    block.pop("note", None)
    return block


def _mask_block(target: Target) -> dict:
    return {
        "radius": _setup._mask_radius_for(target.dataset_class, target.instrument),
        "over_sampling": _OVER_SAMPLE_RECIPE,
    }


def _dataset_hashes(dataset_path: Path) -> dict[str, str | None]:
    dataset_path = Path(dataset_path)
    return {name: _sha256_file(dataset_path / name) for name in _DATASET_FILES}


def _adapt_image_source_block(target: Target, dataset_path: Path) -> str | None:
    """Identity tag for the cached adapt image, or ``None`` for non-pixelized targets."""
    if target.model_type not in _setup._PIX_MODEL_TYPES:
        return None
    cache_path = Path(dataset_path) / "lensed_source.fits"
    digest = _sha256_file(cache_path)
    if digest is None:
        return "not_yet_cached"
    return f"lensed_source.fits:sha256:{digest}"


def _canonical_target_dict(
    target: Target, model: Any, dataset_path: Path, positions_setup: dict | None = None
) -> dict:
    return {
        "cell": target.cell,
        "model_dim": int(model.prior_count),
        "parameter_names": list(model.model_component_and_parameter_names),
        "priors": [_canonical_prior(p) for p in model.priors_ordered_by_id],
        "likelihood": _likelihood_block(target),
        "positions": _positions_block(target, positions_setup),
        "dataset_hashes": _dataset_hashes(dataset_path),
        "mask": _mask_block(target),
        "adapt_image_source": _adapt_image_source_block(target, dataset_path),
        # Every TARGETS-registry entry is a primary (non-diagnostic) target —
        # mirrors the target_class taxonomy in _setup.py's
        # apply_diagnostic_prior_overrides / positions_settings (class 3 =
        # target-changing diagnostic arm); registry targets are class 1.
        "target_class": 1,
    }


def target_id(
    target: Target, model: Any, dataset_path: Path, positions_setup: dict | None = None
) -> str:
    """A short, stable, cross-process identifier for one (target, model, dataset) triple.

    Two calls with structurally identical inputs — same cell, same free
    parameters and priors, same likelihood settings, same positions config,
    same dataset files on disk, same mask/over-sampling recipe, same cached
    adapt image — hash identically, in any process, on any machine. Changing
    any of those changes the id; changing the sampler, seed, n_live, or any
    other pure-algorithm knob does NOT.

    ``positions_setup`` is the resolved positions configuration the run used
    (``_setup.positions_settings()``'s dict, recorded verbatim in the results
    JSON's top-level ``"positions"``). It is a plain argument, never read
    from the environment here, so the id stays recomputable from a recorded
    artifact in any process — see :func:`_positions_block`.
    """
    canonical = _canonical_target_dict(target, model, dataset_path, positions_setup)
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def _module_sha() -> str:
    """Short content hash of this file, for ``target_block``'s ``priors_ref``."""
    return _sha256_file(Path(__file__))[:12]


def target_block(
    target: Target, model: Any, dataset_path: Path, positions_setup: dict | None = None
) -> dict:
    """The schema-v2 ``target`` block for one resolved (target, model, dataset).

    See PROGRAMME.md §5 "Benchmark & result schema (v2)" for the field
    reference; ``target_class_vs_v1`` is always ``null`` here — every
    ``TARGETS``-registry entry is new in schema v2, so none descends from a
    v1 target by construction.
    """
    return {
        "target_id": target_id(target, model, dataset_path, positions_setup),
        "cell": target.cell,
        "model_dim": int(model.prior_count),
        "priors_ref": f"_targets.py@{_module_sha()}",
        "likelihood": _likelihood_block(target),
        "positions": _positions_block(target, positions_setup),
        "tolerances": dataclasses.asdict(target.tolerances),
        "target_class_vs_v1": None,
    }


# -----------------------------------------------------------------------------
# TOLERANCES.md rendering (Step 7) — idempotent: pure function of TARGETS.
# -----------------------------------------------------------------------------


def render_tolerances_markdown() -> str:
    lines = [
        "# Target tolerances",
        "",
        "Auto-generated by `python -m searches._targets --render` "
        "(`scripts/misc/searches/_targets.py`'s `TARGETS` registry, W4 / issue "
        "#161 Phase 1). Do not hand-edit — edit the registry and re-render.",
        "",
        'Pass/fail metrics per PROGRAMME.md §"Phase 1": correctness is '
        "Δ(max logL) vs the truth bar within `delta_max_ll_nats`; posterior "
        "agreement is a per-parameter mean shift within `mean_shift_sigma` and "
        "a σ ratio inside `sigma_ratio`, both measured against a reference "
        "posterior (`results/baselines/InferenceRefs_v1/`).",
        "",
        "| Target | Cell | Positions | Precision | Δmax logL (nats) | "
        "Mean shift (σ) | σ ratio | Notes |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for key in sorted(TARGETS):
        t = TARGETS[key]
        tol = t.tolerances
        lines.append(
            f"| `{key}` | `{t.cell}` | {t.positions} | {t.precision} | "
            f"{tol.delta_max_ll_nats:g} | {tol.mean_shift_sigma:g} | "
            f"[{tol.sigma_ratio[0]:g}, {tol.sigma_ratio[1]:g}] | {t.notes or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _tolerances_path() -> Path:
    return _WORKSPACE_ROOT / "results" / "notes" / "inference" / "targets" / "TOLERANCES.md"


def _render_and_write() -> Path:
    out_path = _tolerances_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_tolerances_markdown())
    return out_path


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="Targets registry (W4 / issue #161, Phase 1 targets registry)."
    )
    _parser.add_argument(
        "--render",
        action="store_true",
        help="Write results/notes/inference/targets/TOLERANCES.md from the registry.",
    )
    _args = _parser.parse_args()
    if _args.render:
        _path = _render_and_write()
        print(f"Wrote {_path}")
    else:
        print(f"{len(TARGETS)} targets registered. Pass --render to write TOLERANCES.md.")
