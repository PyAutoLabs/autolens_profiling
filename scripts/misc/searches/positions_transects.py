"""Phase 4 Stage 1 — PositionsLH eager value_and_grad transects (issue #159).

Characterises the shape ``al.PositionsLH`` adds to the ``imaging/mge/hst``
likelihood surface, *before* any campaign spends compute on it (Phase 4
Stage 1, ``results/notes/inference/PROGRAMME.md``): the threshold hinge
(continuous value, discontinuous derivative), the zero-gradient interior
plateau, and the argmax-switch kink (which pair of the truth's multiple
images attains ``max_sep`` can change discontinuously as the mass model
moves), plus penalty-factor (1e5 / 1e8) and threshold (0.3 fixed vs
SLaM-style ``auto``) sensitivity.

Model on ``clipper_campaign.py`` / ``multi_start_nan_accounting_overhead.py``:
this is a **pure-evaluation** script (no ``search.fit()``), reusing
``searches._setup.build_for_cell`` for the dataset/model/analysis and the
``model.instance_from_vector(vector=..., xp=jnp)`` +
``analysis.log_likelihood_function`` idiom those scripts already use for
eager ``jax.value_and_grad``.

Positions are the simulator's own **truth** positions
(``dataset/imaging/hst/positions.json``, 4 quad images) — an idealisation
(real-data positions are hand-drawn from a completed fit), not a
representativeness claim; see the RESULTS.md caveats.

Usage::

    JAX_ENABLE_X64=True python scripts/misc/searches/positions_transects.py --quick
    JAX_ENABLE_X64=True python scripts/misc/searches/positions_transects.py --n 601

Outputs -> ``results/notes/inference/phase_04_positions/transects/``:
``transect_a.json`` (+ ``.png``), ``transect_b.json`` (+ ``.png``),
``transect_c.json`` (+ ``.png``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
for _p in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from searches._setup import build_for_cell  # noqa: E402

DATASET_CLASS = "imaging"
MODEL_TYPE = "mge"
INSTRUMENT = "hst"
DATASET_PATH = _ROOT / "dataset" / "imaging" / INSTRUMENT

OUTPUT_DIR = _ROOT / "results" / "notes" / "inference" / "phase_04_positions" / "transects"

# Threshold x factor grid (Phase 4 Stage 1 sensitivity arms). Factors are
# (value, label) so the arm name is always "f1e5"/"f1e8" — Python's ``:g``
# format renders 1e5 as "100000" but 1e8 as "1e+08" (inconsistent, and
# doesn't match _setup.py's positions_arm_tag() convention), so the label is
# carried explicitly rather than re-derived from the float.
_THRESHOLD_MODES = ("fixed", "auto")
_FACTORS = ((1e5, "1e5"), (1e8, "1e8"))


def _arm_name(threshold_mode: str, factor_label: str) -> str:
    t = "0.3" if threshold_mode == "fixed" else "auto"
    return f"t{t}_f{factor_label}"


# -----------------------------------------------------------------------------
# Cell / positions / anchor construction
# -----------------------------------------------------------------------------


def build_cell():
    """imaging/mge/hst, positions OFF (the arm swap happens by mutating
    ``analysis.positions_likelihood_list`` per arm below)."""
    dataset, model, analysis = build_for_cell(
        dataset_class=DATASET_CLASS,
        model_type=MODEL_TYPE,
        instrument=INSTRUMENT,
        use_jax=True,
        use_mixed_precision=False,
    )
    assert analysis.positions_likelihood_list is None
    return dataset, model, analysis


def truth_tracer():
    return al.from_json(file_path=DATASET_PATH / "tracer.json")


def truth_positions():
    """The 4 quad-image truth positions (``dataset/imaging/hst/positions.json``,
    derived once and committed by ``_setup._truth_positions_for`` — see
    scripts/misc/searches/README.md's "Position likelihood" section)."""
    positions = al.from_json(file_path=DATASET_PATH / "positions.json")
    assert len(positions) >= 2
    return positions


def build_arms(positions, tracer) -> dict:
    """Resolve the {threshold in (0.3, auto)} x {factor in (1e5, 1e8)} arms.

    ``auto`` replicates SLaM's ``result.positions_threshold_from(factor=3.0,
    minimum_threshold=0.2)`` — see ``_setup.py``'s positions-plumbing section.
    Recorded here rather than imported so this script self-documents the
    formula it is exercising.
    """
    positions_fit = al.SourceMaxSeparation(
        data=positions, noise_map=None, tracer=tracer, plane_redshift=None
    )
    max_sep_truth = float(np.nanmax(positions_fit.max_separation_of_plane_positions))
    auto_threshold = max(3.0 * max_sep_truth, 0.2)

    arms = {}
    for mode in _THRESHOLD_MODES:
        threshold = 0.3 if mode == "fixed" else auto_threshold
        for factor, factor_label in _FACTORS:
            name = _arm_name(mode, factor_label)
            arms[name] = {
                "threshold_mode": mode,
                "threshold": threshold,
                "factor": factor,
            }
    return arms, max_sep_truth, auto_threshold


def make_positions_lh(positions, arm_cfg) -> al.PositionsLH:
    return al.PositionsLH(
        positions=positions,
        threshold=arm_cfg["threshold"],
        log_likelihood_penalty_factor=arm_cfg["factor"],
    )


def _prior_index(model, predicate) -> int:
    """The single ``unique_prior_paths`` index matching ``predicate(path)``.

    ``model.unique_prior_paths`` is index-aligned with the free-parameter
    vector ``vector_from_unit_vector`` / ``instance_from_vector`` consume
    (verified: 15 free params for imaging/mge/hst, path[-1] names matching
    ``model.parameter_names`` 1:1) — this is the general, Gaussian-count-
    independent alternative to hardcoding an index.
    """
    matches = [i for i, path in enumerate(model.unique_prior_paths) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one matching free parameter, found {len(matches)}: "
            f"{[model.unique_prior_paths[i] for i in matches]}"
        )
    return matches[0]


def truth_anchor_vector(model, tracer) -> np.ndarray:
    """A full 15-dim parameter vector for imaging/mge/hst with the lens
    mass + shear (and, where structurally meaningful, the MGE bulges'
    centre/ell_comps) set to the simulator's truth (``tracer.json``).

    Every free parameter here is *structural* (centre / ell_comps /
    einstein_radius / gamma) — the MGE basis's per-Gaussian intensities are
    linear and solved by the inversion inside ``fit_from``, not part of this
    vector at all — so unlike a naive "truth instance" there is no light-
    amplitude mismatch to paper over. Reuses the ``_runner._truth_anchor_for_cell``
    *idea* (truth-derived, not prior-median, starting point) generalised via
    ``unique_prior_paths`` pattern matching rather than a per-dataset-class
    truth dict, since imaging/mge has no existing truth-anchor helper.
    """
    lens = min(tracer.galaxies, key=lambda g: float(g.redshift))
    source = max(tracer.galaxies, key=lambda g: float(g.redshift))
    truth_lookup = {
        ("lens", "bulge", "centre_0"): float(lens.bulge.centre[0]),
        ("lens", "bulge", "centre_1"): float(lens.bulge.centre[1]),
        ("lens", "bulge", "ell_comps_0"): float(lens.bulge.ell_comps[0]),
        ("lens", "bulge", "ell_comps_1"): float(lens.bulge.ell_comps[1]),
        ("lens", "mass", "centre_0"): float(lens.mass.centre[0]),
        ("lens", "mass", "centre_1"): float(lens.mass.centre[1]),
        ("lens", "mass", "ell_comps_0"): float(lens.mass.ell_comps[0]),
        ("lens", "mass", "ell_comps_1"): float(lens.mass.ell_comps[1]),
        ("lens", "mass", "einstein_radius"): float(lens.mass.einstein_radius),
        ("lens", "shear", "gamma_1"): float(lens.shear.gamma_1),
        ("lens", "shear", "gamma_2"): float(lens.shear.gamma_2),
        ("source", "bulge", "centre_0"): float(source.bulge.centre[0]),
        ("source", "bulge", "centre_1"): float(source.bulge.centre[1]),
        ("source", "bulge", "ell_comps_0"): float(source.bulge.ell_comps[0]),
        ("source", "bulge", "ell_comps_1"): float(source.bulge.ell_comps[1]),
    }
    physical = np.asarray(
        model.vector_from_unit_vector([0.5] * model.prior_count), dtype=float
    )
    unmatched = 0
    for i, path in enumerate(model.unique_prior_paths):
        key = (path[1], path[2], path[-1])
        if key in truth_lookup:
            physical[i] = truth_lookup[key]
        else:
            unmatched += 1
    assert unmatched == 0, f"{unmatched} free parameter(s) had no truth mapping"
    return physical


# -----------------------------------------------------------------------------
# Scalar-sweep machinery (one free-parameter index varied, rest held at anchor)
# -----------------------------------------------------------------------------


def make_scalar_fns(model, analysis, anchor_vector, idx: int):
    """``(value_and_grad(logl), value_and_grad(penalty))``, both
    ``jax.jit(jax.vmap(...))`` over the scalar being varied at ``idx``.

    ``analysis.positions_likelihood_list`` must already be set (or ``None``)
    for the arm being measured *before* calling this — each arm gets its own
    freshly-jitted closure baking in that arm's ``PositionsLH`` (or none).

    **Cache these** (see ``get_scalar_fns``): every call here builds a BRAND
    NEW ``jax.jit(...)`` object, and JAX's compile cache is keyed by function
    *identity*, not by whether an equivalent function was compiled before —
    so calling this repeatedly for the same ``(idx, arm)`` (e.g. once per
    Transect-B crossing) pays a full XLA recompile of the real MGE+inversion
    likelihood every time, which is the dominant cost of this script.
    """
    import jax
    import jax.numpy as jnp

    anchor = jnp.asarray(anchor_vector)

    def logl(x):
        vector = anchor.at[idx].set(x)
        instance = model.instance_from_vector(vector=vector, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    def penalty(x):
        vector = anchor.at[idx].set(x)
        instance = model.instance_from_vector(vector=vector, xp=jnp)
        return analysis.log_likelihood_penalty_from(instance=instance)

    return (
        jax.jit(jax.vmap(jax.value_and_grad(logl))),
        jax.jit(jax.vmap(jax.value_and_grad(penalty))),
    )


def get_scalar_fns(cache: dict, model, analysis, anchor_vector, idx: int, arm_name: str, positions, arm_cfg):
    """Cached ``make_scalar_fns`` keyed by ``(idx, arm_name)``.

    JAX's own per-jit-object shape cache still applies on top of this: the
    SAME cached ``(logl_fn, penalty_fn)`` pair is reused across every grid
    shape a caller throws at it (Transect A's coarse grid, Transect B's fine
    windows, the plateau grid, ...), paying exactly one XLA compile per
    distinct (idx, arm, grid-length) rather than per call.
    """
    key = (idx, arm_name)
    # ALWAYS (re)set positions_likelihood_list right before returning --
    # not only on a cache miss. jax.jit traces LAZILY on first call, not at
    # wrap time: resetting this flag immediately after building the jit
    # object (the previous version of this function) meant the eventual
    # first call traced with whatever positions_likelihood_list happened to
    # be current at THAT moment -- often a different arm's config, or None.
    # A cache HIT can also trigger a fresh trace (JAX retraces per distinct
    # input shape, e.g. Transect A's coarse grid vs Transect B's fine
    # window), so this must run unconditionally, not just when building.
    analysis.positions_likelihood_list = (
        None if arm_name == "off" else [make_positions_lh(positions, arm_cfg)]
    )
    if key not in cache:
        cache[key] = make_scalar_fns(model, analysis, anchor_vector, idx)
    return cache[key]


def max_sep_and_argmax(model, analysis, anchor_vector, idx: int, grid: np.ndarray):
    """Arm-independent geometry: ``furthest_separations_of_plane_positions``
    (per truth position) at every grid point, traced through the model's own
    ``tracer_via_instance_from`` — a plain NumPy loop (no grad needed here;
    only ``penalty``/``logl`` need differentiating).

    Returns ``(max_sep[grid], seps[grid, n_positions])``; the argmax pair at
    a point is ``np.flatnonzero(np.isclose(seps[i], max_sep[i]))``.
    """
    positions = truth_positions()
    max_sep = np.empty(len(grid))
    seps_all = np.empty((len(grid), len(positions)))
    for i, x in enumerate(grid):
        vector = anchor_vector.copy()
        vector[idx] = x
        instance = model.instance_from_vector(vector=vector)
        tracer = analysis.tracer_via_instance_from(instance=instance)
        fit = al.SourceMaxSeparation(
            data=positions, noise_map=None, tracer=tracer, plane_redshift=None
        )
        seps = np.asarray(fit.furthest_separations_of_plane_positions.array)
        seps_all[i] = seps
        max_sep[i] = float(np.max(seps))
    return max_sep, seps_all


def argmax_switch_locations(grid: np.ndarray, seps_all: np.ndarray, max_sep: np.ndarray):
    """Grid indices where the SET of positions attaining ``max_sep`` changes."""
    pair_sets = [
        frozenset(np.flatnonzero(np.isclose(seps_all[i], max_sep[i], rtol=1e-9, atol=1e-9)).tolist())
        for i in range(len(grid))
    ]
    switches = []
    for i in range(1, len(grid)):
        if pair_sets[i] != pair_sets[i - 1]:
            switches.append(
                {
                    "x_before": float(grid[i - 1]),
                    "x_after": float(grid[i]),
                    "pair_before": sorted(pair_sets[i - 1]),
                    "pair_after": sorted(pair_sets[i]),
                }
            )
    return switches


# -----------------------------------------------------------------------------
# Chunked vmap evaluation
# -----------------------------------------------------------------------------

# A single jax.vmap over the whole Transect-A grid (601 points) at once tries
# to allocate ~30 GB and OOMs -- this batches the REAL MGE+inversion
# likelihood (15361 masked pixels), not a toy function. `vram.vmap_batch_for`
# (autolens_profiling's own A100 vmap probe) caps this exact cell
# (imaging/mge/hst) at 64; that value is reused here as a CPU-safe default
# too (validated empirically: unchunked quick-mode calls up to ~51 points
# never OOM'd). Padding every chunk to a FIXED size also means each (idx,
# arm) pair needs exactly ONE XLA compile — not one per distinct grid
# length thrown at it across Transects A/B/C.
_VMAP_CHUNK = 64


def chunked_call(fn, grid: np.ndarray, batch_size: int = _VMAP_CHUNK):
    """Call a ``jax.jit(jax.vmap(...))``-wrapped ``fn`` over ``grid`` in
    fixed-size chunks (padding the final chunk by repeating its last point,
    then trimming the padding back off). Returns ``(value, grad)`` as NumPy
    arrays, concatenated back to ``len(grid)``.
    """
    n = len(grid)
    if n <= batch_size:
        value, grad = fn(grid)
        return np.asarray(value), np.asarray(grad)
    values, grads = [], []
    for start in range(0, n, batch_size):
        chunk = np.asarray(grid[start : start + batch_size])
        pad = batch_size - len(chunk)
        if pad > 0:
            chunk = np.concatenate([chunk, np.full(pad, chunk[-1])])
        value, grad = fn(chunk)
        value, grad = np.asarray(value), np.asarray(grad)
        if pad > 0:
            value, grad = value[: batch_size - pad], grad[: batch_size - pad]
        values.append(value)
        grads.append(grad)
    return np.concatenate(values), np.concatenate(grads)


# -----------------------------------------------------------------------------
# Transect A — theta_E in [0, 3]
# -----------------------------------------------------------------------------


def run_transect_a(model, analysis, anchor_vector, arms, positions, n: int, cache: dict) -> dict:
    idx = _prior_index(model, lambda p: p[-1] == "einstein_radius" and p[2] == "mass")
    grid = np.linspace(0.0, 3.0, n)

    max_sep, seps_all = max_sep_and_argmax(model, analysis, anchor_vector, idx, grid)
    argmax_switches = argmax_switch_locations(grid, seps_all, max_sep)

    arm_results = {}
    for arm_name in ("off", *arms):
        arm_cfg = arms.get(arm_name)
        logl_fn, penalty_fn = get_scalar_fns(
            cache, model, analysis, anchor_vector, idx, arm_name, positions, arm_cfg
        )
        logl, dlogl = chunked_call(logl_fn, grid)
        penalty, dpenalty = chunked_call(penalty_fn, grid)
        arm_results[arm_name] = {
            "logl": np.asarray(logl).tolist(),
            "dlogl": np.asarray(dlogl).tolist(),
            "penalty": np.asarray(penalty).tolist(),
            "dpenalty": np.asarray(dpenalty).tolist(),
        }
    analysis.positions_likelihood_list = None

    return {
        "theta_e_grid": grid.tolist(),
        "max_sep": max_sep.tolist(),
        "argmax_switches": argmax_switches,
        "arms": arm_results,
        "einstein_radius_index": idx,
    }


# -----------------------------------------------------------------------------
# Transect B — threshold-crossing fine sweep
# -----------------------------------------------------------------------------


def _find_crossings(grid: np.ndarray, max_sep: np.ndarray, threshold: float) -> list[float]:
    diff = max_sep - threshold
    crossings = []
    for i in range(1, len(grid)):
        if diff[i - 1] == 0.0 or (diff[i - 1] < 0.0) != (diff[i] < 0.0):
            # Linear-interpolate the crossing x, then clip to the grid span.
            if diff[i] == diff[i - 1]:
                x_cross = grid[i - 1]
            else:
                frac = -diff[i - 1] / (diff[i] - diff[i - 1])
                x_cross = grid[i - 1] + frac * (grid[i] - grid[i - 1])
            crossings.append(float(x_cross))
    return crossings


def run_transect_b(
    model, analysis, anchor_vector, arms, positions, transect_a: dict, quick: bool, cache: dict
) -> dict:
    idx = transect_a["einstein_radius_index"]
    coarse_grid = np.asarray(transect_a["theta_e_grid"])
    coarse_max_sep = np.asarray(transect_a["max_sep"])

    fine_step = 1.0e-4
    fine_half_width = 0.05
    n_fine = int(round(2 * fine_half_width / fine_step)) + 1
    if quick:
        # Coarser fine-grid for --quick: same half-width, 100x sparser step,
        # so the crossing region is still resolved but the run is fast.
        n_fine = max(51, n_fine // 100)

    windows = {}
    all_argmax_switches = list(transect_a["argmax_switches"])  # already found on the coarse grid

    for arm_name, cfg in arms.items():
        crossings = _find_crossings(coarse_grid, coarse_max_sep, cfg["threshold"])
        arm_windows = []
        for x_cross in crossings:
            fine_grid = np.linspace(x_cross - fine_half_width, x_cross + fine_half_width, n_fine)
            fine_grid = fine_grid[(fine_grid >= 0.0) & (fine_grid <= 3.0)]
            if len(fine_grid) < 3:
                continue

            max_sep_fine, seps_fine = max_sep_and_argmax(model, analysis, anchor_vector, idx, fine_grid)
            all_argmax_switches.extend(argmax_switch_locations(fine_grid, seps_fine, max_sep_fine))

            logl_fn, penalty_fn = get_scalar_fns(
                cache, model, analysis, anchor_vector, idx, arm_name, positions, cfg
            )
            logl, dlogl = chunked_call(logl_fn, fine_grid)
            penalty, dpenalty = chunked_call(penalty_fn, fine_grid)

            penalty_np = np.asarray(penalty)
            dpenalty_np = np.asarray(dpenalty)

            # Re-locate the crossing on the FINE grid's own (precise)
            # max_sep_fine rather than trusting the coarse-grid-interpolated
            # x_cross to land within one fine-grid step of the true
            # crossing: the coarse Transect-A grid is far too sparse for
            # that (a coarse spacing of ~0.075 vs a fine step of ~0.002-0.05
            # can easily place BOTH probed neighbours on the same side of
            # the true crossing, silently reporting a zero hinge jump).
            fine_crossings = _find_crossings(fine_grid, max_sep_fine, cfg["threshold"])
            resolved = len(fine_crossings) > 0
            x_cross_fine = (
                min(fine_crossings, key=lambda c: abs(c - x_cross)) if resolved else x_cross
            )
            cross_idx = int(np.argmin(np.abs(fine_grid - x_cross_fine)))
            left = max(cross_idx - 1, 0)
            right = min(cross_idx + 1, len(fine_grid) - 1)

            arm_windows.append(
                {
                    "x_cross": x_cross,
                    "x_cross_fine": x_cross_fine,
                    "resolved_in_fine_window": resolved,
                    "fine_grid": fine_grid.tolist(),
                    "max_sep": max_sep_fine.tolist(),
                    "logl": np.asarray(logl).tolist(),
                    "dlogl": np.asarray(dlogl).tolist(),
                    "penalty": penalty_np.tolist(),
                    "dpenalty": dpenalty_np.tolist(),
                    "hinge": {
                        "dpenalty_left": float(dpenalty_np[left]),
                        "dpenalty_right": float(dpenalty_np[right]),
                        "jump": float(dpenalty_np[right] - dpenalty_np[left]),
                        "penalty_left": float(penalty_np[left]),
                        "penalty_right": float(penalty_np[right]),
                        "value_continuity_gap": float(abs(penalty_np[right] - penalty_np[left])),
                    },
                }
            )
        windows[arm_name] = arm_windows

    # Interior plateau: a stretch deep inside every arm's OWN threshold
    # (anchor theta_E=1.6, where max_sep ~ 0 by truth-position construction)
    # must show EXACTLY zero penalty and zero dpenalty throughout. The window
    # must stay clear of that arm's OWN crossings -- a fixed [1.5, 1.7] for
    # every arm is not safe: the tighter 'auto' threshold (0.2 here) crosses
    # at ~1.50/1.70, right at that window's edges. Derive a window inset from
    # each arm's own detected crossings instead (falling back to a narrow
    # +/-0.03 default when no crossing was found on the coarse grid).
    plateau = {}
    for arm_name, cfg in arms.items():
        crossings = _find_crossings(coarse_grid, coarse_max_sep, cfg["threshold"])
        below = [c for c in crossings if c < 1.6]
        above = [c for c in crossings if c > 1.6]
        margin = 0.02
        lo = (max(below) + margin) if below else 1.57
        hi = (min(above) - margin) if above else 1.63
        assert lo < hi, f"[{arm_name}] no safe interior window: crossings {crossings}"
        plateau_grid = np.linspace(lo, hi, 21 if not quick else 5)
        _, penalty_fn = get_scalar_fns(
            cache, model, analysis, anchor_vector, idx, arm_name, positions, cfg
        )
        penalty, dpenalty = chunked_call(penalty_fn, plateau_grid)
        plateau[arm_name] = {
            "grid": plateau_grid.tolist(),
            "penalty_max_abs": float(np.max(np.abs(np.asarray(penalty)))),
            "dpenalty_max_abs": float(np.max(np.abs(np.asarray(dpenalty)))),
            "is_exact_zero_plateau": bool(
                np.all(np.asarray(penalty) == 0.0) and np.all(np.asarray(dpenalty) == 0.0)
            ),
        }

    return {
        "windows": windows,
        "argmax_switches_all": all_argmax_switches,
        "plateau": plateau,
    }


# -----------------------------------------------------------------------------
# Transect C — ell_comps / shear at fixed theta_E, + gradient-norm ratio
# -----------------------------------------------------------------------------


def make_full_grad_fns(model, analysis, anchor_vector, idx: int):
    """``jax.jit(jax.vmap(jax.grad(...)))`` of ``logl``/``penalty`` wrt the
    FULL 15-dim parameter vector, batched over ``vector[idx] = x`` for an
    array of ``x`` — for the gradient-norm-ratio measurement (how much the
    penalty term dominates the search direction). Batching + jitting this
    (rather than one un-jitted ``jax.grad`` call per sample point) is what
    keeps the 5-points-per-arm-per-sweep sensitivity check cheap.
    """
    import jax
    import jax.numpy as jnp

    anchor = jnp.asarray(anchor_vector)

    def logl(x):
        vector = anchor.at[idx].set(x)
        instance = model.instance_from_vector(vector=vector, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    def penalty(x):
        vector = anchor.at[idx].set(x)
        instance = model.instance_from_vector(vector=vector, xp=jnp)
        return analysis.log_likelihood_penalty_from(instance=instance)

    return jax.jit(jax.vmap(jax.grad(logl))), jax.jit(jax.vmap(jax.grad(penalty)))


def get_full_grad_fns(cache: dict, model, analysis, anchor_vector, idx: int, arm_name: str, positions, arm_cfg):
    """Cached ``make_full_grad_fns``, mirroring ``get_scalar_fns`` — same
    "never rebuild a jit object you're about to call again" rationale."""
    key = ("full_grad", idx, arm_name)
    # See get_scalar_fns's docstring/comment: this must be set unconditionally
    # (not only on cache miss), since JAX may retrace a cached jit object at
    # a new input shape.
    analysis.positions_likelihood_list = (
        None if arm_name == "off" else [make_positions_lh(positions, arm_cfg)]
    )
    if key not in cache:
        cache[key] = make_full_grad_fns(model, analysis, anchor_vector, idx)
    return cache[key]


def run_transect_c(model, analysis, anchor_vector, arms, positions, n: int, cache: dict) -> dict:
    idx_ell0 = _prior_index(
        model, lambda p: p[1] == "lens" and p[2] == "mass" and p[-1] == "ell_comps_0"
    )
    idx_gamma1 = _prior_index(model, lambda p: p[2] == "shear" and p[-1] == "gamma_1")

    sweeps = {}
    for label, idx, lo, hi in (
        ("mass_ell_comps_0", idx_ell0, -0.3, 0.3),
        ("shear_gamma_1", idx_gamma1, -0.2, 0.2),
    ):
        grid = np.linspace(lo, hi, n)
        arm_results = {}
        for arm_name in ("off", *arms):
            arm_cfg = arms.get(arm_name)
            logl_fn, penalty_fn = get_scalar_fns(
                cache, model, analysis, anchor_vector, idx, arm_name, positions, arm_cfg
            )
            logl, dlogl = chunked_call(logl_fn, grid)
            penalty, dpenalty = chunked_call(penalty_fn, grid)
            arm_results[arm_name] = {
                "logl": np.asarray(logl).tolist(),
                "dlogl": np.asarray(dlogl).tolist(),
                "penalty": np.asarray(penalty).tolist(),
                "dpenalty": np.asarray(dpenalty).tolist(),
            }

        # Gradient-norm ratio at 5 sample points across the sweep. A full
        # 15-dim jax.grad of the real MGE+inversion likelihood is its own
        # fresh XLA compile per (idx, arm) — expensive (tens of seconds each)
        # even batched via vmap over the 5 points — so this is measured for
        # ONE representative "on" arm per sweep (the fixed-threshold,
        # factor=1e8 arm; threshold/factor sensitivity is already the whole
        # point of Transects A/B) rather than repeating it across all 4.
        sample_idx = np.linspace(0, n - 1, 5).astype(int)
        sample_x = grid[sample_idx]
        ratio_arm_name = _arm_name("fixed", "1e8")
        ratio_results = {}
        for arm_name, cfg in ((ratio_arm_name, arms[ratio_arm_name]),):
            grad_logl_fn, grad_penalty_fn = get_full_grad_fns(
                cache, model, analysis, anchor_vector, idx, arm_name, positions, cfg
            )
            grad_logl_batch = np.asarray(grad_logl_fn(sample_x))
            grad_penalty_batch = np.asarray(grad_penalty_fn(sample_x))
            rows = []
            for j, x_value in enumerate(sample_x):
                norm_logl = float(np.linalg.norm(grad_logl_batch[j]))
                norm_penalty = float(np.linalg.norm(grad_penalty_batch[j]))
                ratio = norm_penalty / norm_logl if norm_logl > 0.0 else None
                rows.append(
                    {
                        "x": float(x_value),
                        "grad_norm_logl": norm_logl,
                        "grad_norm_penalty": norm_penalty,
                        "ratio_penalty_over_logl": ratio,
                    }
                )
            ratio_results[arm_name] = rows

        sweeps[label] = {"grid": grid.tolist(), "arms": arm_results, "gradient_norm_ratio": ratio_results}

    return sweeps


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_transect_a(transect_a: dict, path: Path) -> None:
    grid = np.asarray(transect_a["theta_e_grid"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for arm_name, res in transect_a["arms"].items():
        axes[0].plot(grid, res["logl"], label=arm_name, linewidth=1.0)
        axes[1].plot(grid, res["dlogl"], label=arm_name, linewidth=1.0)
    axes[0].set_ylabel("log_likelihood_function")
    axes[1].set_ylabel("d(logl)/d(theta_E)")
    axes[1].set_xlabel("theta_E")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].set_title("Transect A: imaging/mge/hst, theta_E in [0, 3]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_transect_b(transect_b: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for arm_name, windows in transect_b["windows"].items():
        for w in windows:
            ax.plot(w["fine_grid"], w["dpenalty"], label=f"{arm_name} @ {w['x_cross']:.3f}", linewidth=1.0)
            plotted = True
    if plotted:
        ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel("theta_E (fine window around threshold crossing)")
    ax.set_ylabel("d(penalty)/d(theta_E)")
    ax.set_title("Transect B: threshold-crossing hinge (one-sided gradient jump)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_transect_c(transect_c: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(transect_c), figsize=(6 * len(transect_c), 5))
    if len(transect_c) == 1:
        axes = [axes]
    for ax, (label, sweep) in zip(axes, transect_c.items()):
        grid = np.asarray(sweep["grid"])
        for arm_name, res in sweep["arms"].items():
            ax.plot(grid, res["penalty"], label=arm_name, linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("penalty")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=601, help="Transect A / C grid points (default 601).")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small grids (n=61) and a sparse Transect-B fine grid, for fast iteration.",
    )
    args = parser.parse_args()

    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "positions_transects.py requires fp64 — run with JAX_ENABLE_X64=True."
        )

    n = 41 if args.quick else args.n
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- positions_transects.py [n={n}, quick={args.quick}] ---")
    print("Building imaging/mge/hst (positions OFF)...")
    _dataset, model, analysis = build_cell()

    tracer = truth_tracer()
    positions = truth_positions()
    arms, max_sep_truth, auto_threshold = build_arms(positions, tracer)
    print(f"  truth positions: {len(positions)} points; max_sep(truth)={max_sep_truth:.3e}")
    print(f"  auto threshold = max(3.0 * {max_sep_truth:.3e}, 0.2) = {auto_threshold:.6f}")
    print(f"  arms: {list(arms)}")

    anchor_vector = truth_anchor_vector(model, tracer)
    print(f"  anchor vector (truth-mapped, {model.prior_count} free params): {anchor_vector.tolist()}")

    # Shared across A/B/C: caches every (idx, arm) jit(vmap(...)) object so
    # it is compiled ONCE and reused for every grid shape asked of it — see
    # get_scalar_fns / get_full_grad_fns docstrings. Without this, every
    # Transect-B crossing / plateau check / Transect-C sample point would
    # pay a fresh XLA recompile of the real MGE+inversion likelihood.
    fn_cache: dict = {}

    print("\nTransect A: theta_E in [0, 3]...")
    transect_a = run_transect_a(model, analysis, anchor_vector, arms, positions, n, fn_cache)
    (OUTPUT_DIR / "transect_a.json").write_text(
        json.dumps(
            {
                "dataset_class": DATASET_CLASS,
                "model_type": MODEL_TYPE,
                "instrument": INSTRUMENT,
                "arms_config": arms,
                "max_sep_truth": max_sep_truth,
                "auto_threshold": auto_threshold,
                "anchor_vector": anchor_vector.tolist(),
                **transect_a,
            },
            indent=2,
        )
    )
    plot_transect_a(transect_a, OUTPUT_DIR / "transect_a.png")
    print(f"  {len(transect_a['argmax_switches'])} argmax-switch(es) found on the coarse grid.")
    print("  wrote transect_a.json / transect_a.png")

    print("\nTransect B: threshold-crossing fine sweep (+/- 0.05)...")
    transect_b = run_transect_b(
        model, analysis, anchor_vector, arms, positions, transect_a, quick=args.quick, cache=fn_cache
    )
    (OUTPUT_DIR / "transect_b.json").write_text(
        json.dumps({"arms_config": arms, **transect_b}, indent=2)
    )
    plot_transect_b(transect_b, OUTPUT_DIR / "transect_b.png")
    for arm_name, windows in transect_b["windows"].items():
        for w in windows:
            h = w["hinge"]
            print(
                f"  [{arm_name}] crossing @ theta_E={w['x_cross_fine']:.5f} "
                f"(resolved_in_fine_window={w['resolved_in_fine_window']}): "
                f"dpenalty jump={h['jump']:.6e}  value-continuity gap={h['value_continuity_gap']:.3e}"
            )
    print(f"  {len(transect_b['argmax_switches_all'])} argmax-switch(es) found (coarse + fine).")
    for arm_name, p in transect_b["plateau"].items():
        print(f"  [{arm_name}] interior plateau (theta_E in [1.5,1.7]) exact-zero={p['is_exact_zero_plateau']}")
    print("  wrote transect_b.json / transect_b.png")

    print("\nTransect C: ell_comps / shear at fixed theta_E...")
    n_c = 21 if args.quick else 121
    transect_c = run_transect_c(model, analysis, anchor_vector, arms, positions, n_c, fn_cache)
    (OUTPUT_DIR / "transect_c.json").write_text(json.dumps({"arms_config": arms, **transect_c}, indent=2))
    plot_transect_c(transect_c, OUTPUT_DIR / "transect_c.png")
    for label, sweep in transect_c.items():
        for arm_name, rows in sweep["gradient_norm_ratio"].items():
            ratios = [r["ratio_penalty_over_logl"] for r in rows if r["ratio_penalty_over_logl"] is not None]
            if ratios:
                print(f"  [{label}/{arm_name}] grad-norm ratio (penalty/logl) range: "
                      f"{min(ratios):.3e} - {max(ratios):.3e}")
    print("  wrote transect_c.json / transect_c.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
