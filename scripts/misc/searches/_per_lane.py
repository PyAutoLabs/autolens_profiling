"""Per-lane diagnostics for the ``af.MultiStart*`` gradient searches (CP-3).

The inference programme's Phase 3 question is a **per-start basin-hit
probability**, not a per-run success flag (``PROGRAMME.md`` §2.3), and its
experimental rules require that "every lane/chain/particle-set final *and
best* state survives into artifacts. No winner-only records." (§3). The
ordinary ``searches/`` results JSON carries neither: it records the winner
(``max_log_likelihood``) and nothing about the other ``n_starts - 1`` lanes.

This module is the minimal bridge. It captures the search's own
``search_internal`` dict as it is written and reshapes it into a per-lane
block for the results JSON.

Two facts force the capture-as-written approach (both learned the hard way by
``clipper_campaign.py``, whose docstring records them):

- ``search_internal`` is **deleted on successful completion**, so the raw dict
  cannot be read back off disk after ``fit()`` returns.
- ``save_search_internal`` must be patched at **class** level: ``fit()``
  rebuilds ``search.paths``, so a hook attached to the instance before the fit
  is silently discarded.

What is recorded, per lane, and where it comes from:

===========================  =================================================
``final_params``             ``search_internal["params"]`` — the lane's FINAL
                             position, PHYSICAL (the search writes it back in
                             physical units even when it stepped in scaled
                             ones).
``lane_best_params``         ``search_internal["lane_best_params"]`` — the
                             lane's own BEST position (PyAutoFit PR#1515).
``lane_best_fom``            ``search_internal["lane_best_foms"]``.
``lane_best_step``           ``search_internal["lane_best_steps"]`` — indexes
                             into ``fom_history``.
``basin``                    Always ``None``. Basin classification is a
                             human-adjudicated threshold decision; this module
                             records the raw data it will be computed from and
                             deliberately bakes in no truth bar.
===========================  =================================================

**There is no per-lane FINAL figure of merit anywhere.** ``fom_history`` is the
GLOBAL best-fom trace (one scalar per step), and
``samples_via_internal_from`` writes ``np.nan`` for every non-winning start's
log likelihood. So ``final_fom`` is recorded as ``None`` — a lane's *best* fom
is the only per-lane objective value that exists. That is precisely why the
per-lane-best change was CP-3's library pre-requisite.

A resurrected lane is a NEW start: PyAutoFit resets its record to
``NaN`` params / ``inf`` fom, so such a lane reads as
``lane_best_fom: null`` here rather than as a lane that found nothing.

H3.3 (``ell_comps`` trapping) accounting is raw counts only:

- ``n_pinned_final`` / ``n_pinned_best`` — how many of the lane's parameters
  sit exactly on a prior-box bound, at its final position and at its best one.
  The reference box is ``ClipperPriorBox.bounds_from_model``: the margin-INSET
  box the clipper actually projects onto, which is where a clipped lane parks.
  The full box is recorded alongside the per-lane vectors so any other bound
  convention can be recomputed from the artifact.
- ``ell_comps_magnitude_final`` / ``_best`` — the magnitude
  ``sqrt(e1**2 + e2**2)`` of every ``ell_comps`` pair in the model. The corner
  region (both components strictly inside the box, magnitude above the
  ``ELL_COMPS_MAGNITUDE_CLAMP`` plateau) versus the clamp annulus is the
  localisation Phase 3 asks for — but the split is left to the reader: the
  magnitudes are recorded, not thresholded.
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


from contextlib import contextmanager  # noqa: E402
from typing import Any  # noqa: E402

import autofit as af  # noqa: E402
import numpy as np  # noqa: E402
from autofit.non_linear.paths.directory import DirectoryPaths  # noqa: E402

# Counters the search writes into ``search_internal``. Read with ``.get`` so a
# missing key stays ``None``: `0` and `null` are different findings (a `0` says
# the search watched and saw nothing; a `null` says it never wrote the key, i.e.
# broken plumbing or an older PyAutoFit).
_COUNTER_KEYS = (
    "total_steps",
    "stop_reason",
    "best_fom",
    "n_resurrections",
    "n_value_nan_lane_steps",
    "n_grad_nan_lane_steps",
    "n_constrained_lane_steps",
    "n_clipped_lane_steps",
)


@contextmanager
def capture_search_internal():
    """Yield a dict that fills with the search's ``search_internal`` payload.

    Patches ``DirectoryPaths.save_search_internal`` at CLASS level for the
    duration of the block (see the module docstring for why the instance is
    not enough) and always restores it, including on an exception.

    The dict is *updated* on every write, so after a fit it holds the LAST
    checkpoint the search wrote — which for a completed run is the terminal
    one built immediately before ``_fit`` returns.
    """
    captured: dict = {}
    real = DirectoryPaths.save_search_internal

    def spy(self, obj):
        captured.update(obj)
        return real(self, obj)

    DirectoryPaths.save_search_internal = spy
    try:
        yield captured
    finally:
        DirectoryPaths.save_search_internal = real


def _as_list(value: Any) -> list | None:
    """JSON-safe list from an array-like, ``None`` when absent.

    Non-finite entries become ``None`` rather than the bare ``NaN`` /
    ``Infinity`` tokens ``json.dumps`` emits, which are not valid JSON and
    silently break strict parsers.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return [None if not np.isfinite(v) else float(v) for v in array.ravel()]


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _ell_comps_pairs(names: list[str]) -> dict[str, tuple[int, int]]:
    """Map ``<component path>.ell_comps`` -> the (index_0, index_1) pair.

    Keyed on the dotted component path so two profiles' ``ell_comps`` (lens
    mass vs shear vs an MGE basis) stay distinct — a short-name key would
    collapse them into one entry and lose most of the trapping evidence.
    """
    halves: dict[str, dict[str, int]] = {}
    for index, name in enumerate(names):
        if "ell_comps" not in name:
            continue
        for suffix in ("_0", "_1"):
            if name.endswith(suffix):
                halves.setdefault(name[: -len(suffix)], {})[suffix] = index
    return {
        stem: (pair["_0"], pair["_1"])
        for stem, pair in halves.items()
        if "_0" in pair and "_1" in pair
    }


def _magnitudes(vector: np.ndarray | None, pairs: dict[str, tuple[int, int]]) -> dict | None:
    if vector is None:
        return None
    out = {}
    for stem, (i, j) in pairs.items():
        out[stem] = _finite_or_none(float(np.hypot(vector[i], vector[j])))
    return out


def _pinned_mask(
    vector: np.ndarray | None, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray | None:
    """Boolean mask of coordinates sitting exactly on a (finite) box bound.

    ``rtol=0``: the question is "exactly on the bound", so the tolerance is
    absolute and scaled to the bound's own magnitude. (``clipper_campaign.py``
    left ``np.isclose``'s default ``rtol=1e-5`` in play and derived its atol
    from ``upper`` only; that is a looser test than this one, so pinning counts
    here are not directly comparable to that campaign's ``lanes_pinned``.)

    A ``NaN`` coordinate — a lane resurrected after death — is never pinned.
    """
    if vector is None:
        return None
    atol_lower = 1e-9 + 1e-6 * np.abs(np.where(np.isfinite(lower), lower, 0.0))
    atol_upper = 1e-9 + 1e-6 * np.abs(np.where(np.isfinite(upper), upper, 0.0))
    on_lower = np.isfinite(lower) & np.isclose(vector, lower, rtol=0.0, atol=atol_lower)
    on_upper = np.isfinite(upper) & np.isclose(vector, upper, rtol=0.0, atol=atol_upper)
    return (on_lower | on_upper) & np.isfinite(vector)


def _row(vector: np.ndarray | None) -> np.ndarray | None:
    """A finite-or-NaN float row, or ``None`` when the array is absent."""
    if vector is None:
        return None
    return np.asarray(vector, dtype=float)


def per_lane_block(
    *,
    captured: dict,
    model: Any,
    n_starts: int,
) -> dict:
    """Build the ``diagnostics`` block for one MultiStart* run.

    ``captured`` is the dict filled by :func:`capture_search_internal`. Every
    field is read defensively: an empty capture (the search never wrote a
    checkpoint) yields a block whose ``valid`` is ``False`` with the reason
    attached, rather than an exception that would discard a completed fit.
    """
    names = list(model.model_component_and_parameter_names)
    lower, upper = af.ClipperPriorBox().bounds_from_model(model=model)
    pairs = _ell_comps_pairs(names)

    finals = _row(captured.get("params"))
    bests = _row(captured.get("lane_best_params"))
    best_foms = _row(captured.get("lane_best_foms"))
    best_steps = captured.get("lane_best_steps")
    best_steps = None if best_steps is None else np.asarray(best_steps)

    problems: list[str] = []
    if not captured:
        problems.append(
            "no search_internal was captured: the search wrote no checkpoint, so "
            "every per-lane field below is null and this run cannot be interpreted"
        )
    if bests is None:
        problems.append(
            "search_internal has no 'lane_best_params': the installed PyAutoFit "
            "predates the per-lane-best change (PR#1515), so per-lane basin "
            "classification is not possible from this artifact"
        )
    n_lanes = None if finals is None else int(finals.shape[0])
    if n_lanes is not None and n_lanes != n_starts:
        # The signature of a completed-fit resume collision: `fit()` returned a
        # CACHED result from a differently-configured arm without entering
        # `_fit`, so these counters describe a different run.
        problems.append(
            f"captured {n_lanes} lanes but the search was configured with "
            f"n_starts={n_starts}: this run returned another arm's result "
            f"(completed-fit resume collision) or the capture is stale"
        )

    lanes = []
    for i in range(n_lanes or 0):
        final_row = finals[i]
        best_row = None if bests is None else bests[i]
        if best_row is not None and not np.isfinite(best_row).any():
            # Resurrected-after-death slot: PyAutoFit blanks the record to NaN.
            best_row = None
        pinned_final = _pinned_mask(final_row, lower, upper)
        pinned_best = _pinned_mask(best_row, lower, upper)
        lanes.append(
            {
                "lane": i,
                "final_params": _as_list(final_row),
                # No per-lane final objective exists anywhere in the search's
                # records — see the module docstring. Recorded as an explicit
                # null so a reader is not left inferring it was forgotten.
                "final_fom": None,
                "lane_best_params": _as_list(best_row),
                "lane_best_fom": None if best_foms is None else _finite_or_none(best_foms[i]),
                # Fitness.call returns -2 * log_posterior.
                "lane_best_log_posterior": (
                    None
                    if best_foms is None or not np.isfinite(best_foms[i])
                    else -0.5 * float(best_foms[i])
                ),
                "lane_best_step": (
                    None if best_steps is None else int(np.asarray(best_steps).ravel()[i])
                ),
                "n_pinned_final": None if pinned_final is None else int(pinned_final.sum()),
                "n_pinned_best": None if pinned_best is None else int(pinned_best.sum()),
                "pinned_final_names": (
                    None
                    if pinned_final is None
                    else [names[j] for j in np.flatnonzero(pinned_final)]
                ),
                "pinned_best_names": (
                    None if pinned_best is None else [names[j] for j in np.flatnonzero(pinned_best)]
                ),
                "ell_comps_magnitude_final": _magnitudes(final_row, pairs),
                "ell_comps_magnitude_best": _magnitudes(best_row, pairs),
                # Adjudicated by a human against the target's truth bar, later.
                # Never written here: a threshold baked in at record time is a
                # verdict masquerading as data.
                "basin": None,
            }
        )

    alive_history = captured.get("alive_history")
    alive = None if alive_history is None else np.asarray(alive_history).astype(int).tolist()
    fom_history = captured.get("fom_history")

    block: dict = {
        "n_starts_configured": int(n_starts),
        "n_lanes_recorded": n_lanes,
        "parameter_names": names,
        "prior_box": {
            # The clipper's margin-INSET box, i.e. what a clipped lane parks on.
            "clipper_lower": _as_list(lower),
            "clipper_upper": _as_list(upper),
            # The raw prior limits, so the inset is visible rather than implied.
            "prior_lower": [
                _finite_or_none(getattr(p, "lower_limit", None)) for p in model.priors_ordered_by_id
            ],
            "prior_upper": [
                _finite_or_none(getattr(p, "upper_limit", None)) for p in model.priors_ordered_by_id
            ],
        },
        "ell_comps_pairs": {stem: list(idx) for stem, idx in pairs.items()},
        "counters": {key: captured.get(key) for key in _COUNTER_KEYS},
        # Per-step histories, not lifetime totals (PROGRAMME.md §3): the lane
        # counters above are survival integrals and cannot answer "when".
        "alive_history": alive,
        "fom_history_global_best": _as_list(fom_history),
        "per_lane": lanes,
        "valid": not problems,
        "invalid_reasons": problems,
    }
    return block
