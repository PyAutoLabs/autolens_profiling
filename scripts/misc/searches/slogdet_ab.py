"""Phase 8A — slogdet vs cholesky A/B on the free-AdaptSplit regularization NaN wall.

Critical-path item CP-4 of the inference programme
(``results/notes/inference/PROGRAMME.md`` §4 "Phase 8 — 8A", §9 "CP-4"). Under
free ``AdaptSplit`` regularization the curvature-plus-regularization matrix goes
non-positive-definite at high coefficients — the doubly-squared lambda^4
fragility of #104 — and the Cholesky returns NaN there (JAX backend), a wall a
gradient search cannot step through. PyAutoArray#391 shipped an **opt-in,
non-default** ``Settings(log_det_method="slogdet")`` covering both evidence
log-det terms. This script measures whether turning it on removes the wall
without moving the target. See "WHICH CELL ACTUALLY WALLS" below: the mesh the
reg is paired with decides whether the high-coefficient region is a NaN wall or
merely an over-regularized floor.

PRE-REGISTERED PASS CRITERIA (all four must hold; stated before any run):

    1. zero slogdet NaNs on the replay set
    2. value equality within float tolerance on PD points
    3. finite gradients
    4. runtime within 2x

Criterion 2 is the category-1/2 claim: where the Cholesky SUCCEEDED the two
methods are computing the same number, so the target has not moved; the only
places the arms may differ are the points where the Cholesky failed, and those
deltas are quantified as **category 2** (equivalent-within-tolerance) rather
than waved through (``PROGRAMME.md`` §3, "Smoothing taxonomy enforced at record
level").

That criterion assumes the mutually-finite draws are one population. A 14-draw
laptop probe of ``delaunay_adapt_split`` says they are not: away from the wall
the two methods agree to ~4e-6 absolute at |logL| ~ 1.6e5 (relative ~3e-11),
but in a band just before the Cholesky NaNs — coefficients 1e2 to 5e4 in that
probe — they separate by O(0.05-0.15) nats while the Cholesky is still
returning a number. So the verdict splits the comparison into ``clean_pd`` and
``marginal_band``, reporting each with the coefficient range that produced it.
Pooling them yields one max delta that describes neither.

Criteria 1 and 3 are scored on the treatment arm ALONE, exactly as registered.
A draw that is NaN, or whose gradient is non-finite, under BOTH arms therefore
fails them even though ``log_det_method`` cannot be the cause. The criteria are
not softened for that; instead the verdict carries an ``attribution`` block
recording how the control did on the same draws, so a failure can be read as
"slogdet broke this" or "the control broke it too" without re-litigating the
pre-registration.

WHICH CELL ACTUALLY WALLS — read before choosing ``--model-type``
-----------------------------------------------------------------
The pre-registration names the stressor as "the free-AdaptSplit (knn) target".
This repo's own #117 record separates two pairings that the phrase conflates,
and only one of them is a NaN wall:

- ``--model-type knn`` — KNN mesh + free AdaptSplit. Its high-coefficient region
  is an over-regularized floor: bad, but **finite** and escapable by
  resurrection at ~step 1300 (``_setup._knn_model``). This is the cell the
  pre-registration names.
- ``--model-type delaunay_adapt_split`` — Delaunay mesh + free AdaptSplit. This
  is the **NaN wall**: the #104 doubly-squared lambda^4 fragility, lanes dying
  instead of learning, a ~2000-step resurrection lottery to escape the
  +8.5k-logL plateau (``_setup._delaunay_adapt_split_model``, and
  ``scripts/imaging/searches/multi_start_prodigy/delaunay.py``).

Run BOTH. The ``knn`` arm is what the programme registered and it is the cell
whose gradient work slogdet would be recommended for; the
``delaunay_adapt_split`` arm is where the failure slogdet exists to remove has
actually been observed, so it is the arm that can produce a non-vacuous
criterion-1 result. A ``knn`` run that reports zero cholesky NaNs is a real
finding about the knn cell — not a failed experiment — but it cannot on its own
support "slogdet removes the AdaptSplit NaN wall".

WHY THIS SCRIPT HAS A HARVEST STAGE
-----------------------------------
Phase 8A is written as "replay the recorded rejected draws". **There are no
recorded rejected draws.** Nothing in this repo persists a parameter vector at
the moment a likelihood returned NaN: the hazard records under
``results/hazards/`` hold deterministic probe grids, and the NaN-counter harvest
from RAL jobs 335003-5
(``results/notes/inference/phase_00_unblocking/ral_harvest/nan_check_335003_5/``)
holds counters, a wall time and one formatted best-fit string — no vectors. The
one place per-draw records exist is PyAutoFit's own ``search_internal``, which a
completed search deletes. So the replay set has to be **generated**, and this
script generates it once and writes it to disk so every later phase replays the
SAME draws rather than re-deriving its own.

Draw sources (each row is tagged with its ``source``, so the verdict can be read
per source rather than pooled):

``prior``
    Seeded uniform unit-cube draws mapped through the model's own priors — the
    broad-start population a MultiStartGradient search actually begins from.

``truth_bar``
    Gaussian perturbations, in unit-cube coordinates, around an anchor vector —
    the "truth-bar region" of the pre-registration. NOTE the anchor defaults to
    the prior medians because **the knn truth-basin vector is not recorded
    anywhere in this repo**; the #117 campaign reports only its scalars (max
    logL +29724 at r_E 1.599). Pass ``--anchor-json`` with a
    ``{parameter_name: value}`` mapping to centre the region on the real truth
    basin; the resolved anchor and its provenance are written into the artifact
    either way, so a run centred on prior medians cannot be mistaken for one
    centred on truth.

``lambda_transect``
    The anchor vector with the free ``AdaptSplit`` coefficients swept
    log-uniformly across their full prior support (1e-6 … 1e6). This is the
    **positive control**: the wall is a conditioning effect in the coefficients,
    so a transect spanning six decades either side of unity must cross it. If
    the cholesky arm produces zero NaNs on this source the run is VOID — it has
    not exercised the thing it exists to test — and the script exits non-zero
    rather than reporting a clean pass. That check matters here specifically:
    the CPU-tier arms of jobs 335003-5 measured **zero** value-NaN lane-steps
    (``phase_00_unblocking/RESULTS.md:189``), i.e. the search-trajectory wall is
    so far a GPU-scale observation, and a CPU replay that quietly found nothing
    to fix would otherwise read as a pass.

``descent`` (optional, ``--descent-steps N``)
    Real search-trajectory draws: a short ``af.MultiStartProdigy`` run under the
    CHOLESKY arm, with ``save_search_internal`` spied at CLASS level (an
    instance-level hook is discarded — ``fit()`` rebuilds ``search.paths``) and
    every checkpointed lane-parameter array appended. This is a
    checkpoint-cadence SUBSAMPLE of the lane trajectories, not every evaluated
    draw; PyAutoFit does not retain a per-step parameter history. Off by default
    because it costs a real search.

WHAT MAKES THIS AN HONEST A/B
-----------------------------
Both arms are built by ``_setup.build_ab_for_cell``, which constructs one
dataset, one model and one ``AdaptImages`` and hands the same objects to both
analyses; the arms differ in the single string ``al.Settings(log_det_method=)``
and in nothing else. The artifact records a ``target_common_id`` — a hash of the
target block with ``log_det_method`` removed — and the run ASSERTS it is
identical across arms, so "the arms differ only in log_det_method" is a checked
property of the record rather than a claim in a docstring.

Both arms then evaluate the SAME draws array loaded from the SAME file, in the
same order, at the same batch size, on the same device, in fp64.

The objective is the log-likelihood alone, not the search's
``-(log_l + log_p)``. The log-prior term is analytic and cannot be touched by a
log-det method, so folding it in would only add a constant to both arms and
dilute the gradient-finiteness measurement.

Usage (from the ``autolens_profiling/`` root)::

    python3 scripts/misc/searches/slogdet_ab.py --stage all
    python3 scripts/misc/searches/slogdet_ab.py --stage harvest --descent-steps 60
    python3 scripts/misc/searches/slogdet_ab.py --stage replay

Writes:

    results/notes/inference/phase_08_regularization/slogdet_ab/draws/<cell>.npz
    results/notes/inference/phase_08_regularization/slogdet_ab/<cell>_<hardware>.json
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


import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402
from autofit.non_linear.paths.directory import DirectoryPaths  # noqa: E402
from autonerves import conf  # noqa: E402
from searches._setup import build_ab_for_cell  # noqa: E402

# The two arms. "cholesky" is the CONTROL and is named explicitly rather than
# left as None: an arm whose method is implicit reads the config, and a config
# override anywhere in the chain would silently change what the control is.
CONTROL = "cholesky"
TREATMENT = "slogdet"

# The free AdaptSplit coefficients. These are the parameters whose conditioning
# produces the wall, and the transect sweeps them.
_REG_COEFFICIENT_NAMES = ("inner_coefficient", "outer_coefficient")

# batch_size=4 is MANDATORY on pixelized cells: the unbatched 16-wide jvp fusion
# is the ~58 GB allocation that killed the #117 runs (see
# scripts/imaging/searches/multi_start_prodigy/knn.py). The default here matches
# the framework's own pixelized default in searches/_samplers.py.
DEFAULT_BATCH_SIZE = 4


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", default="all", choices=("harvest", "replay", "all"))
    p.add_argument("--dataset-class", default="imaging")
    p.add_argument(
        "--model-type",
        default="knn",
        help="knn = the cell the pre-registration names (free AdaptSplit on the "
        "KNN mesh, whose high-coefficient region is a finite floor); "
        "delaunay_adapt_split = the cell where the NaN wall was actually "
        "observed. Run both — see the module docstring.",
    )
    p.add_argument("--instrument", default="hst")
    p.add_argument("--seed", type=int, default=0)

    # --- draw-set composition (harvest stage) ---
    p.add_argument("--n-prior", type=int, default=64, help="broad-start prior draws")
    p.add_argument("--n-truth-bar", type=int, default=64, help="draws around the anchor")
    p.add_argument(
        "--truth-bar-width",
        type=float,
        default=0.05,
        help="sigma of the unit-cube Gaussian around the anchor",
    )
    p.add_argument(
        "--transect-points",
        type=int,
        default=64,
        help="log-spaced AdaptSplit coefficient values across the prior support",
    )
    p.add_argument(
        "--anchor-json",
        default=None,
        help="JSON file of {parameter_name: physical_value} overriding the "
        "prior-median anchor; use it to centre the truth-bar region on the "
        "real truth basin once that vector is known",
    )
    p.add_argument(
        "--descent-steps",
        type=int,
        default=0,
        help="if >0, also harvest checkpointed lane vectors from a "
        "MultiStartProdigy run of this many steps (costs a real search)",
    )
    p.add_argument("--descent-starts", type=int, default=16)

    # --- replay stage ---
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--value-atol",
        type=float,
        default=1e-6,
        help="absolute tolerance for value equality on mutually-PD points",
    )
    p.add_argument(
        "--value-rtol",
        type=float,
        default=1e-9,
        help="relative tolerance for value equality on mutually-PD points",
    )
    p.add_argument(
        "--runtime-factor",
        type=float,
        default=2.0,
        help="pre-registered runtime ceiling: slogdet wall / cholesky wall",
    )
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--tag", default="", help="free-form tag recorded in the artifact")
    p.add_argument(
        "--draws-tag",
        default="",
        help="namespace suffix for the draw-set file. Two runs at different "
        "draw-set sizes (a CPU pass and an A100 pass, say) MUST use different "
        "tags: without one the second harvest silently overwrites the first's "
        "npz while the first's JSON keeps pointing at the path, and the two "
        "artifacts then disagree about which draws produced them.",
    )
    p.add_argument("--out-dir", default=None, help="override the results directory")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Paths / labels
# -----------------------------------------------------------------------------


def cell_name(args) -> str:
    return f"{args.dataset_class}_{args.model_type}_{args.instrument}"


def results_dir(args) -> _Path:
    if args.out_dir:
        return _Path(args.out_dir)
    return _ROOT / "results" / "notes" / "inference" / "phase_08_regularization" / "slogdet_ab"


def draws_path(args) -> _Path:
    suffix = f"_{args.draws_tag}" if args.draws_tag else ""
    return results_dir(args) / "draws" / f"{cell_name(args)}{suffix}.npz"


def hardware_label(jax) -> str:
    kind = jax.default_backend()
    if kind == "gpu":
        return f"gpu_{jax.devices()[0].device_kind.replace(' ', '_')}"
    return f"cpu_{platform.machine()}"


# -----------------------------------------------------------------------------
# Target identity
# -----------------------------------------------------------------------------


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def target_block(args, model, log_det_method: str) -> dict:
    """The §5-schema ``target`` block for one arm.

    ``target_common`` is everything EXCEPT ``log_det_method``; hashing it
    separately is what lets the run prove the two arms share a target rather
    than assert it.
    """
    common = {
        "cell": f"{args.dataset_class}/{args.model_type}/{args.instrument}",
        "model_dim": int(model.prior_count),
        "parameter_names": list(model.model_component_and_parameter_names),
        "priors": [repr(prior) for prior in model.priors_ordered_by_id],
        "likelihood": {
            "positive_only": bool(al.Settings().use_positive_only_solver),
            "border_relocator": True,
            "curvature_floor": float(al.Settings().no_regularization_add_to_curvature_diag_value),
            "regularization_term_method": al.Settings().regularization_term_method,
            "precision": "mp" if args.mixed_precision else "fp64",
        },
        "positions": {"enabled": False},
    }
    common_id = _hash(common)
    block = dict(common)
    block["likelihood"] = dict(common["likelihood"], log_det_method=log_det_method)
    block["target_common_id"] = common_id
    block["target_id"] = _hash(block)
    # Category 2 relative to the control arm: the treatment is claimed
    # equivalent-within-tolerance, and the replay is what tests that claim.
    block["target_class_vs_control"] = None if log_det_method == CONTROL else 2
    return block


# -----------------------------------------------------------------------------
# Harvest — build the replay set
# -----------------------------------------------------------------------------


def anchor_vector(args, model) -> tuple[np.ndarray, str]:
    """The centre of the truth-bar region and the base of the transect.

    Returns ``(vector, provenance)``. Defaults to prior medians and SAYS SO —
    the knn truth-basin vector is not recorded in this repo, and a run centred
    on prior medians must not be readable as one centred on truth.
    """
    names = list(model.model_component_and_parameter_names)
    vector = np.asarray(model.vector_from_unit_vector([0.5] * model.prior_count), dtype=float)
    if args.anchor_json is None:
        return vector, "prior_medians (no truth-basin vector is recorded in-repo)"

    overrides = json.loads(_Path(args.anchor_json).read_text())
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ValueError(
            f"--anchor-json names parameters absent from this model: {unknown}. "
            f"Known names: {names}"
        )
    for name, value in overrides.items():
        vector[names.index(name)] = float(value)
    return vector, f"{args.anchor_json} ({len(overrides)} overrides on prior medians)"


def _coefficient_indices(model) -> list[int]:
    names = list(model.model_component_and_parameter_names)
    idx = [i for i, name in enumerate(names) if name.split(".")[-1] in _REG_COEFFICIENT_NAMES]
    if not idx:
        raise RuntimeError(
            f"no free AdaptSplit coefficients found in this model "
            f"({_REG_COEFFICIENT_NAMES}); the lambda transect — the positive "
            f"control of this experiment — cannot be built, so the run would "
            f"have no way to prove it reached the wall. Parameter names: {names}"
        )
    return idx


def _coefficient_support(model, indices: list[int]) -> tuple[float, float]:
    priors = list(model.priors_ordered_by_id)
    lowers = [float(priors[i].lower_limit) for i in indices]
    uppers = [float(priors[i].upper_limit) for i in indices]
    return min(lowers), max(uppers)


def harvest(args, model) -> dict:
    """Build the draw set. Likelihood-free except for the optional descent arm."""
    rng = np.random.default_rng(args.seed)
    ndim = int(model.prior_count)
    anchor, anchor_provenance = anchor_vector(args, model)

    vectors: list[np.ndarray] = []
    sources: list[str] = []

    # --- prior: the broad-start population ---
    for _ in range(args.n_prior):
        unit = rng.uniform(size=ndim)
        vectors.append(np.asarray(model.vector_from_unit_vector(list(unit)), dtype=float))
        sources.append("prior")

    # --- truth_bar: a ball around the anchor, in UNIT-CUBE coordinates ---
    #
    # Perturbing in unit-cube space rather than physical space keeps the ball
    # commensurate with each prior's own width; a fixed physical sigma would be
    # a rounding error on a LogUniform coefficient spanning twelve decades and a
    # wild jump on a centre coordinate.
    anchor_unit = _unit_vector_of(model, anchor)
    for _ in range(args.n_truth_bar):
        unit = np.clip(anchor_unit + rng.normal(scale=args.truth_bar_width, size=ndim), 0.0, 1.0)
        vectors.append(np.asarray(model.vector_from_unit_vector(list(unit)), dtype=float))
        sources.append("truth_bar")

    # --- lambda_transect: the positive control ---
    indices = _coefficient_indices(model)
    lower, upper = _coefficient_support(model, indices)
    for lam in np.logspace(np.log10(lower), np.log10(upper), args.transect_points):
        vector = anchor.copy()
        for i in indices:
            vector[i] = float(lam)
        vectors.append(vector)
        sources.append("lambda_transect")

    record = {
        "vectors": np.asarray(vectors, dtype=float),
        "sources": np.asarray(sources),
        "anchor": anchor,
        "anchor_provenance": anchor_provenance,
        "parameter_names": np.asarray(list(model.model_component_and_parameter_names)),
        "seed": args.seed,
        "coefficient_indices": np.asarray(indices),
        "transect_support": np.asarray([lower, upper]),
    }
    return record


def _unit_vector_of(model, vector: np.ndarray) -> np.ndarray:
    """Invert ``vector_from_unit_vector`` prior-by-prior.

    ``af.AbstractPriorModel`` has no vectorised inverse, so this goes through
    each prior's own ``unit_value_for``/``value_for`` pair. Done explicitly
    rather than approximated, because a wrong inverse would silently move the
    truth-bar ball somewhere else entirely.
    """
    priors = list(model.priors_ordered_by_id)
    out = np.empty(len(priors), dtype=float)
    for i, prior in enumerate(priors):
        out[i] = float(prior.unit_value_for(vector[i]))
    return np.clip(out, 0.0, 1.0)


def harvest_descent(args, model, analysis, record: dict) -> dict:
    """Append checkpointed lane vectors from a short cholesky-arm Prodigy run.

    The spy is attached at CLASS level and APPENDS rather than overwrites: the
    search checkpoints ``search_internal`` repeatedly during the fit, so
    appending yields a trajectory while ``dict.update`` (the pattern in
    ``clipper_campaign.py``, which only wants the final state) yields one point.
    """
    captured: list[np.ndarray] = []
    real = DirectoryPaths.save_search_internal

    def spy(self, obj):
        params = obj.get("params")
        if params is not None:
            captured.append(np.asarray(params, dtype=float).copy())
        lane_best = obj.get("lane_best_params")
        if lane_best is not None:
            captured.append(np.asarray(lane_best, dtype=float).copy())
        return real(self, obj)

    out_root = _ROOT / "output" / "slogdet_ab_harvest"
    # A stale `.completed` short-circuits `fit()` into returning a CACHED result
    # in seconds with no checkpoints written at all — which would look like a
    # descent arm that simply found nothing.
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    conf.instance.output_path = str(out_root)

    search = af.MultiStartProdigy(
        name=f"slogdet_ab_harvest_{cell_name(args)}",
        n_starts=args.descent_starts,
        n_steps=args.descent_steps,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=args.seed,
        number_of_cores=1,
        convergence=af.MultiStartGradientConvergence(check_for_convergence=False),
    )

    DirectoryPaths.save_search_internal = spy
    try:
        search.fit(model=model, analysis=analysis)
    finally:
        DirectoryPaths.save_search_internal = real

    if not captured:
        raise RuntimeError(
            "the descent harvest captured no checkpointed parameter arrays: the "
            "search either short-circuited on a cached result or never reached a "
            "checkpoint boundary. Either way it contributed no draws, so it must "
            "not be recorded as if it had."
        )

    stacked = np.concatenate([c.reshape(-1, c.shape[-1]) for c in captured], axis=0)
    # Duplicate lane states across adjacent checkpoints carry no extra
    # information and would inflate every count in the verdict.
    stacked = np.unique(stacked, axis=0)

    record["vectors"] = np.concatenate([record["vectors"], stacked], axis=0)
    record["sources"] = np.concatenate([record["sources"], np.array(["descent"] * len(stacked))])
    record["descent_steps"] = args.descent_steps
    record["descent_starts"] = args.descent_starts
    return record


# -----------------------------------------------------------------------------
# Replay — evaluate one arm over the draw set
# -----------------------------------------------------------------------------


def build_evaluator(model, analysis):
    """``jit(vmap(value_and_grad(log_likelihood)))`` over a physical vector batch.

    The objective is the log-likelihood ALONE. The search's own figure of merit
    is ``-(log_l + log_p)``, but the log-prior is analytic and untouched by a
    log-det method, so including it would add the same constant to both arms
    while diluting criterion 3 (gradient finiteness) with a term that cannot
    fail.
    """
    import jax
    import jax.numpy as jnp

    def log_likelihood(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    return jax.jit(jax.vmap(jax.value_and_grad(log_likelihood)))


def run_arm(*, label: str, model, analysis, vectors: np.ndarray, batch_size: int) -> dict:
    """Evaluate every draw under one arm. Returns values, gradient finiteness, walls."""
    import jax

    evaluate = build_evaluator(model, analysis)

    n, ndim = vectors.shape
    n_batches = int(np.ceil(n / batch_size))
    padded = np.zeros((n_batches * batch_size, ndim), dtype=float)
    padded[:n] = vectors
    # Pad with a REPEAT of the first row rather than zeros: a zero vector is not
    # a valid model point and could raise inside the likelihood, killing a batch
    # whose real rows were fine.
    padded[n:] = vectors[0]

    # Cold call on a full-width batch, timed separately. Programme rule §3:
    # compile time is recorded on its own and never folded into the algorithmic
    # wall.
    t0 = time.perf_counter()
    jax.block_until_ready(evaluate(padded[:batch_size]))
    compile_s = time.perf_counter() - t0
    print(f"  [{label}] compile + first batch: {compile_s:.1f}s", flush=True)

    values = np.empty(n_batches * batch_size, dtype=float)
    grad_finite = np.empty(n_batches * batch_size, dtype=bool)

    t0 = time.perf_counter()
    for b in range(n_batches):
        lo = b * batch_size
        batch = padded[lo : lo + batch_size]
        value, grad = evaluate(batch)
        jax.block_until_ready((value, grad))
        values[lo : lo + batch_size] = np.asarray(value)
        grad_finite[lo : lo + batch_size] = np.all(np.isfinite(np.asarray(grad)), axis=1)
        if (b + 1) % 10 == 0 or b + 1 == n_batches:
            print(f"  [{label}] batch {b + 1}/{n_batches}", flush=True)
    warm_wall_s = time.perf_counter() - t0

    return {
        "log_det_method": label,
        "values": values[:n],
        "grad_finite": grad_finite[:n],
        "compile_s": compile_s,
        "warm_wall_s": warm_wall_s,
        "s_per_draw": warm_wall_s / n,
        "n_batches": n_batches,
        "batch_size": batch_size,
    }


# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------


def verdict(
    args,
    control: dict,
    treatment: dict,
    sources: np.ndarray,
    vectors: np.ndarray,
    coefficient_indices: np.ndarray,
) -> dict:
    c_val, t_val = control["values"], treatment["values"]
    c_fin, t_fin = np.isfinite(c_val), np.isfinite(t_val)

    both_finite = c_fin & t_fin
    rescued = (~c_fin) & t_fin
    regressed = c_fin & (~t_fin)
    shared_nan = (~c_fin) & (~t_fin)

    delta = np.abs(c_val[both_finite] - t_val[both_finite]) if both_finite.any() else np.array([])
    scale = np.abs(c_val[both_finite]) if both_finite.any() else np.array([])
    rel = delta / np.maximum(scale, 1e-300) if delta.size else np.array([])
    tol = args.value_atol + args.value_rtol * scale if delta.size else np.array([])
    exceed = int(np.count_nonzero(delta > tol)) if delta.size else 0

    # --- the marginal band: mutually-finite draws where the arms DISAGREE ---
    #
    # Criterion 2 assumes the mutually-finite set is one population — "the PD
    # points" — where the two computations must agree. It is not. Approaching
    # the wall there is a band where the Cholesky still returns a number but has
    # already lost digits to the conditioning, and the two methods separate by
    # far more than round-off before the Cholesky finally NaNs. Those draws are
    # what fails criterion 2, and pooling them with the clean PD set hides both:
    # it turns a sharp "equal to 1e-10 away from the wall, O(0.1) beside it"
    # into one meaningless max. So the band is isolated and reported with the
    # coefficient values that produced it — that range IS the finding.
    marginal_global = np.zeros(c_val.size, dtype=bool)
    if delta.size:
        marginal_global[np.flatnonzero(both_finite)[delta > tol]] = True
    clean = both_finite & (~marginal_global)
    clean_delta = np.abs(c_val[clean] - t_val[clean]) if clean.any() else np.array([])
    coefficients = vectors[:, np.asarray(coefficient_indices)] if len(coefficient_indices) else None

    runtime_ratio = treatment["warm_wall_s"] / control["warm_wall_s"]

    # --- the four pre-registered criteria, evaluated verbatim ---
    #
    # Verbatim is the point: these are the criteria the programme registered
    # before the run, and they are scored as written even where a softer reading
    # would be kinder. See `attribution` below for the diagnostics a human needs
    # in order to tell a criterion-3 failure caused by log_det_method from one
    # the control shares.
    criteria = {
        "zero_slogdet_nans": int(np.count_nonzero(~t_fin)) == 0,
        "value_equality_on_pd_points": exceed == 0,
        "finite_gradients": bool(np.all(treatment["grad_finite"])),
        "runtime_within_factor": runtime_ratio <= args.runtime_factor,
    }

    # --- attribution: is a failed criterion the treatment's fault? ---
    #
    # Criterion 3 is scored on the treatment arm alone, so a draw whose gradient
    # is non-finite under BOTH methods fails it — even though log_det_method
    # cannot be the cause. That is not a reason to soften the criterion; it is a
    # reason to record, beside it, the comparison that says whose fault a
    # failure is. Without this the artifact reports "criterion 3 FAILED" and a
    # reader has no way to see that the control failed identically.
    c_grad, t_grad = control["grad_finite"], treatment["grad_finite"]
    attribution = {
        "grad_nonfinite_under_both": int(np.count_nonzero((~c_grad) & (~t_grad))),
        "grad_nonfinite_only_under_slogdet": int(np.count_nonzero(c_grad & (~t_grad))),
        "grad_finite_only_under_slogdet": int(np.count_nonzero((~c_grad) & t_grad)),
        # The adjudicable form of criterion 3: slogdet introduced no new
        # gradient non-finiteness. A verbatim criterion-3 FAIL with this True
        # means the shared mechanism is elsewhere in the likelihood.
        "slogdet_introduces_no_new_grad_nans": int(np.count_nonzero(c_grad & (~t_grad))) == 0,
        # ... and the same for values: a shared NaN is not a slogdet NaN.
        "slogdet_nan_not_shared_with_cholesky": int(np.count_nonzero(c_fin & (~t_fin))),
    }

    # --- validity: the run must have exercised the wall at all ---
    #
    # Recorded as a VOID verdict rather than a pass. A cholesky arm with no NaNs
    # has nothing for slogdet to remove, so all four criteria pass vacuously —
    # which is exactly how a null result gets manufactured. The CPU tier has
    # already measured zero value-NaN lane-steps once
    # (phase_00_unblocking/RESULTS.md:189), so this is a live failure mode, not
    # a theoretical one.
    control_nan_total = int(np.count_nonzero(~c_fin))
    transect = sources == "lambda_transect"
    control_nan_transect = int(np.count_nonzero((~c_fin) & transect))
    problems = []
    if control_nan_total == 0:
        problems.append(
            "the cholesky arm produced ZERO NaNs on the whole replay set: this "
            "run never reached the wall, so its four passes are vacuous. Widen "
            "the transect, raise --transect-points, or move to the hardware tier "
            "where the wall was observed"
        )
    elif control_nan_transect == 0:
        problems.append(
            "the cholesky arm produced NaNs, but NONE on the lambda_transect — "
            "the positive control did not fire, so the NaNs came from somewhere "
            "other than the coefficient conditioning this experiment is about"
        )
    if int(np.count_nonzero(regressed)):
        problems.append(
            f"{int(np.count_nonzero(regressed))} draw(s) are finite under "
            f"cholesky and NaN under slogdet: that is a REGRESSION, not a fix"
        )

    per_source = {}
    for source in sorted(set(sources.tolist())):
        m = sources == source
        per_source[source] = {
            "n": int(np.count_nonzero(m)),
            "cholesky_nan": int(np.count_nonzero((~c_fin) & m)),
            "slogdet_nan": int(np.count_nonzero((~t_fin) & m)),
            "rescued": int(np.count_nonzero(rescued & m)),
            "cholesky_grad_nonfinite": int(np.count_nonzero(~control["grad_finite"] & m)),
            "slogdet_grad_nonfinite": int(np.count_nonzero(~treatment["grad_finite"] & m)),
        }

    return {
        "criteria": criteria,
        "attribution": attribution,
        "pass": all(criteria.values()) and not problems,
        "void": bool(problems),
        "void_reasons": problems,
        "counts": {
            "n_draws": int(c_val.size),
            "cholesky_nan": control_nan_total,
            "cholesky_nan_on_transect": control_nan_transect,
            "slogdet_nan": int(np.count_nonzero(~t_fin)),
            "both_finite": int(np.count_nonzero(both_finite)),
            "rescued_by_slogdet": int(np.count_nonzero(rescued)),
            "regressed_by_slogdet": int(np.count_nonzero(regressed)),
            "nan_under_both": int(np.count_nonzero(shared_nan)),
            "cholesky_grad_nonfinite": int(np.count_nonzero(~control["grad_finite"])),
            "slogdet_grad_nonfinite": int(np.count_nonzero(~treatment["grad_finite"])),
        },
        # Criterion 2. `max_abs_delta` on the mutually-PD set is the
        # category-1 evidence: the target did not move where the Cholesky
        # worked.
        "value_equality": {
            "n_compared": int(delta.size),
            "max_abs_delta": float(delta.max()) if delta.size else None,
            "max_rel_delta": float(rel.max()) if rel.size else None,
            "n_exceeding_tolerance": exceed,
            "atol": args.value_atol,
            "rtol": args.value_rtol,
            # The clean sub-population: mutually finite AND within tolerance.
            # Its max delta is the category-1 number — "away from the wall the
            # two methods compute the same thing to this many digits".
            "clean_pd": {
                "n": int(np.count_nonzero(clean)),
                "max_abs_delta": float(clean_delta.max()) if clean_delta.size else None,
            },
        },
        # The band where the Cholesky still returns a number but has already
        # degraded. Neither a pass nor a regression on its own: it says where
        # the equality claim stops holding, in the coordinate that causes it.
        "marginal_band": {
            "n": exceed,
            "draw_indices": np.flatnonzero(marginal_global).tolist(),
            "max_abs_delta": (
                float(np.abs(c_val[marginal_global] - t_val[marginal_global]).max())
                if marginal_global.any()
                else None
            ),
            "coefficient_min": (
                float(coefficients[marginal_global].min())
                if coefficients is not None and marginal_global.any()
                else None
            ),
            "coefficient_max": (
                float(coefficients[marginal_global].max())
                if coefficients is not None and marginal_global.any()
                else None
            ),
        },
        # Category 2 quantification: what slogdet returns where cholesky failed.
        # These have no cholesky counterpart by construction, so they are
        # reported as a distribution, not as a delta.
        "category_2_rescued_values": {
            "n": int(np.count_nonzero(rescued)),
            "min": float(t_val[rescued].min()) if rescued.any() else None,
            "max": float(t_val[rescued].max()) if rescued.any() else None,
            "median": float(np.median(t_val[rescued])) if rescued.any() else None,
        },
        "runtime": {
            "cholesky_warm_wall_s": control["warm_wall_s"],
            "slogdet_warm_wall_s": treatment["warm_wall_s"],
            "ratio": runtime_ratio,
            "ceiling": args.runtime_factor,
            "cholesky_compile_s": control["compile_s"],
            "slogdet_compile_s": treatment["compile_s"],
        },
        "per_source": per_source,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)

    import jax

    x64 = jax.numpy.zeros(1, dtype=jax.numpy.float64).dtype == np.float64
    if not x64:
        # fp32 manufactures conditioning failures of its own, so a wall measured
        # there would not be the wall this experiment is about — and the
        # cholesky/slogdet delta on the PD set would be dominated by round-off
        # rather than by the two algorithms. Fatal even under
        # --mixed-precision, whose fp32 paths deliberately exclude the log-det.
        raise SystemExit(
            "JAX_ENABLE_X64 is not set. This A/B is an fp64 measurement — run "
            "with JAX_ENABLE_X64=True (every RAL sbatch exports it explicitly)."
        )

    dataset, model, analyses = build_ab_for_cell(
        dataset_class=args.dataset_class,
        model_type=args.model_type,
        instrument=args.instrument,
        use_jax=True,
        use_mixed_precision=args.mixed_precision,
        log_det_methods=(CONTROL, TREATMENT),
    )

    targets = {m: target_block(args, model, m) for m in (CONTROL, TREATMENT)}
    common = {m: t["target_common_id"] for m, t in targets.items()}
    if len(set(common.values())) != 1:
        raise RuntimeError(
            f"the two arms do not share a target: target_common_id={common}. "
            f"Something other than log_det_method differs between them, so this "
            f"is not a category-1/2 A/B and must not be recorded as one."
        )
    if targets[CONTROL]["target_id"] == targets[TREATMENT]["target_id"]:
        raise RuntimeError(
            "both arms hashed to the same target_id — log_det_method did not "
            "enter the hash, so the artifact cannot distinguish them."
        )

    npz = draws_path(args)

    if args.stage in ("harvest", "all"):
        print(f"=== HARVEST — {cell_name(args)} ===", flush=True)
        record = harvest(args, model)
        if args.descent_steps > 0:
            print(
                f"  descent: {args.descent_starts} starts x {args.descent_steps} steps", flush=True
            )
            record = harvest_descent(args, model, analyses[CONTROL], record)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **record)
        print(f"  {len(record['vectors'])} draws -> {npz}", flush=True)

    if args.stage == "harvest":
        return

    if not npz.exists():
        raise SystemExit(
            f"no draw set at {npz}. There are no recorded rejected draws in this "
            f"repo to fall back on — run `--stage harvest` first."
        )

    loaded = np.load(npz, allow_pickle=False)
    vectors = loaded["vectors"]
    sources = loaded["sources"].astype(str)
    if vectors.shape[1] != model.prior_count:
        raise SystemExit(
            f"the stored draw set has {vectors.shape[1]} parameters but this "
            f"model has {model.prior_count}: the draws were harvested against a "
            f"different model and replaying them would compare two targets."
        )

    print(f"=== REPLAY — {len(vectors)} draws, batch_size={args.batch_size} ===", flush=True)
    arms = {}
    for method in (CONTROL, TREATMENT):
        print(f"--- arm: {method} ---", flush=True)
        arms[method] = run_arm(
            label=method,
            model=model,
            analysis=analyses[method],
            vectors=vectors,
            batch_size=args.batch_size,
        )

    result = verdict(
        args,
        arms[CONTROL],
        arms[TREATMENT],
        sources,
        vectors,
        loaded["coefficient_indices"],
    )

    hardware = hardware_label(jax)
    artifact = {
        "schema_version": 2,
        "experiment": "phase_8a_slogdet_ab",
        "critical_path_item": "CP-4",
        "version": al.__version__,
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "tag": args.tag,
        "target_control": targets[CONTROL],
        "target_treatment": targets[TREATMENT],
        "draws": {
            # Repo-relative when it is inside the repo (the committed case, so
            # the artifact does not carry a machine-specific absolute path);
            # absolute when --out-dir put it elsewhere, which is a scratch run.
            "path": (str(npz.relative_to(_ROOT)) if npz.is_relative_to(_ROOT) else str(npz)),
            "n": int(len(vectors)),
            "seed": int(loaded["seed"]),
            "anchor_provenance": str(loaded["anchor_provenance"]),
            "per_source_n": {
                s: int(np.count_nonzero(sources == s)) for s in sorted(set(sources.tolist()))
            },
        },
        "hardware": {
            "tier": hardware,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "x64_enabled": bool(x64),
            "cache_state": "warm (compile timed separately)",
            "compile_s": {m: arms[m]["compile_s"] for m in arms},
        },
        "arms": {
            m: {k: v for k, v in arms[m].items() if k not in ("values", "grad_finite")}
            for m in arms
        },
        "verdict": result,
        "pass_criteria_preregistered": [
            "zero slogdet NaNs on the replay set",
            "value equality within float tolerance on PD points",
            "finite gradients",
            "runtime within 2x",
        ],
    }

    suffix = f"_{args.draws_tag}" if args.draws_tag else ""
    dest = results_dir(args) / f"{cell_name(args)}{suffix}_{hardware}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(artifact, indent=2) + "\n")

    # Per-draw values kept beside the artifact: the JSON carries the verdict,
    # the npz carries the evidence it was computed from, so a reader can
    # recompute the verdict instead of trusting it.
    per_draw = dest.with_suffix(".per_draw.npz")
    np.savez(
        per_draw,
        vectors=vectors,
        sources=sources,
        cholesky_values=arms[CONTROL]["values"],
        slogdet_values=arms[TREATMENT]["values"],
        cholesky_grad_finite=arms[CONTROL]["grad_finite"],
        slogdet_grad_finite=arms[TREATMENT]["grad_finite"],
    )

    _print_verdict(artifact)
    print(f"\nwritten: {dest}")
    print(f"written: {per_draw}")

    if result["void"]:
        raise SystemExit(1)


def _print_verdict(artifact: dict) -> None:
    result = artifact["verdict"]
    counts = result["counts"]
    print("\n" + "=" * 78)
    print("PHASE 8A — slogdet A/B")
    print("=" * 78)
    print(f"draws           {counts['n_draws']}")
    print(
        f"cholesky NaN    {counts['cholesky_nan']}  (transect {counts['cholesky_nan_on_transect']})"
    )
    print(f"slogdet  NaN    {counts['slogdet_nan']}")
    print(f"rescued         {counts['rescued_by_slogdet']}")
    print(f"regressed       {counts['regressed_by_slogdet']}")
    print(
        f"grad non-finite cholesky {counts['cholesky_grad_nonfinite']}  slogdet {counts['slogdet_grad_nonfinite']}"
    )
    eq = result["value_equality"]
    print(
        f"value equality  n={eq['n_compared']} max|Δ|={eq['max_abs_delta']} exceeding={eq['n_exceeding_tolerance']}"
    )
    print(f"  clean PD      n={eq['clean_pd']['n']} max|Δ|={eq['clean_pd']['max_abs_delta']}")
    band = result["marginal_band"]
    print(
        f"  marginal band n={band['n']} max|Δ|={band['max_abs_delta']} "
        f"coeff {band['coefficient_min']} .. {band['coefficient_max']}"
    )
    rt = result["runtime"]
    print(
        f"runtime         cholesky {rt['cholesky_warm_wall_s']:.1f}s  slogdet {rt['slogdet_warm_wall_s']:.1f}s  ratio {rt['ratio']:.2f}x"
    )
    print("\npre-registered criteria:")
    for name, ok in result["criteria"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    attr = result["attribution"]
    print("\nattribution (whose fault a failure is):")
    print(f"  grad non-finite under BOTH arms      {attr['grad_nonfinite_under_both']}")
    print(f"  grad non-finite only under slogdet   {attr['grad_nonfinite_only_under_slogdet']}")
    print(f"  grad rescued by slogdet              {attr['grad_finite_only_under_slogdet']}")
    print(f"  slogdet NaNs cholesky did NOT share  {attr['slogdet_nan_not_shared_with_cholesky']}")
    if result["void"]:
        print("\nVOID — this run cannot speak to the question:")
        for reason in result["void_reasons"]:
            print(f"  - {reason}")
    else:
        print(f"\nVERDICT: {'PASS' if result['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
