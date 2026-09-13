"""The graded Δlog L draw set — construction, and the tables derived from it.

Phase 3 of the ``fixed-lens-light-profiling`` epic (autolens_profiling#255).

Why this module exists
----------------------

Phases 0-2 measured the certified active set at **one** model: the prior-median
instance, which after SLaM ``light[1]`` is essentially the truth. It certifies in
2 passes on Delaunay and 7 on rectangular, and those pass counts *are* the cost
of the scheme. A non-linear search spends almost all of its evaluations far from
the truth, so the production question is not "what does the scheme cost at the
answer" but "what does it cost at a **bad** model, and how often does a fixed
pass budget fail to certify".

Answering that needs a draw set whose badness is *graded and reproducible*: a
set of models sitting at known log-likelihood offsets from the fiducial, built
the same way on every mesh and every device, so the A100 and CPU legs measure
the same models rather than two different bad-model populations.

What a draw is
--------------

A draw is a set of **offsets to the fiducial mass parameters**, nothing more.
Everything downstream — the S0 fit at that mass, the MGE intensities solved
there, the regular-profile conversion, the subtraction, the source-only S3
system — is rebuilt from the offsets exactly as ``fixed_light_probe_cpu.py``
rebuilds its ±1 % draws, through
:func:`active_set_steps.fixed_light_system_from`. A draw is therefore a *model*,
not a perturbed matrix: the lens light re-solves at the wrong mass, which is
what a search actually does.

Two families:

**One-parameter walks.** ``einstein_radius``, ``ell_comps_0``, ``centre_x`` and
the mass **slope**, each scaled by :func:`bisect_to_target` to land at
Δlog L ≈ −10, −100, −1000, −1e4. The achieved Δlog L and the parameter offset
are both recorded — the target is where the bisection *aimed*, never what is
quoted.

**Random draws.** ``n_random`` draws from SLaM-like priors around the truth
(Gaussian, σ = ``RANDOM_SIGMA_SCALE`` × the cell's own prior σ for each of
``einstein_radius``, ``ell_comps`` and ``centre``, fixed seed), so the offsets
are not all one-parameter walks and the joint structure of a real early-search
sample is represented.

The slope walk and the Isothermal fiducial
------------------------------------------

The fiducial mass is ``al.mp.Isothermal``, which has **no slope parameter**. The
slope walk therefore *promotes* the mass to ``al.mp.PowerLaw`` at ``slope=2.0``
— the Isothermal special case — and the cell confirms the promoted fiducial's S0
``figure_of_merit`` matches the Isothermal's to the pinned digits **before**
walking. Without that check a slope row would silently be measuring the
promotion, not the slope.

Δlog L
------

``Δlog L = S0 figure_of_merit(draw) − S0 figure_of_merit(fiducial)``: the whole
library likelihood of the system the cells fit today, at the draw's mass, minus
the same at the fiducial. It is negative by construction (the fiducial is the
best model in the set) and it is measured on **S0**, not S3, because S0 is the
system a search scores — S3 is the re-arrangement whose *solver* this epic is
about.

Everything here is numpy. ``autolens`` is imported lazily inside the two
functions that build profiles, so the tests can exercise the bisection, the
random draws and the derived tables without a JAX or autolens import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# The draw-set definition
# ---------------------------------------------------------------------------

#: The one-parameter walks, in the order they are run and reported.
WALK_PARAMETERS = ("einstein_radius", "ell_comps_0", "centre_x", "slope")

#: Δlog L targets each walk is bisected onto.
DEFAULT_TARGETS = (-10.0, -100.0, -1000.0, -1.0e4)

#: The prior σ of the cells' own model (``build_shared`` / the breakdown cells'
#: PART A). The random draws are Gaussian at ``RANDOM_SIGMA_SCALE`` × these, so
#: they are "SLaM-like priors, widened" rather than an invented scale.
PRIOR_SIGMA = {
    "einstein_radius": 0.05,
    "ell_comps_0": 0.01,
    "ell_comps_1": 0.01,
    "centre_0": 0.005,
    "centre_1": 0.005,
}

#: How much wider than the prior the random draws are. 5× puts the bulk of the
#: sample where an early search actually sits — several σ from the truth — while
#: keeping the draws inside the region where the MGE light solve still converges.
RANDOM_SIGMA_SCALE = 5.0

#: Which parameters the random draws move. The slope is deliberately absent:
#: the random family is the *Isothermal* early-search sample, and promoting it
#: would make the random rows a different model family from the walks.
RANDOM_PARAMETERS = ("einstein_radius", "ell_comps_0", "ell_comps_1", "centre_0", "centre_1")

#: Sign of each walk: which direction away from the fiducial is walked. One
#: direction per parameter, fixed, so a walk row is reproducible — not a choice
#: re-made per run.
WALK_SIGN = {
    "einstein_radius": +1.0,
    "ell_comps_0": +1.0,
    "centre_x": +1.0,
    "slope": +1.0,
}

#: First probe step of each walk, in the parameter's own units. Chosen so the
#: first evaluation lands in the Δlog L ≈ −10³ region measured at the fiducial
#: (θ_E × 1.01 ≈ −1.4e3 on Delaunay), which makes the quadratic seed below
#: informative on the very first call rather than after a doubling ladder.
PROBE_DELTA = {
    "einstein_radius": 0.016,
    "ell_comps_0": 0.01,
    "centre_x": 0.01,
    "slope": 0.02,
}

#: Hard ceiling on each walk's offset. A walk that has to go past this to reach
#: its target has left the region where the model is a *lens* at all, and the
#: row is reported as not reached rather than chased.
MAX_DELTA = {
    "einstein_radius": 1.2,
    "ell_comps_0": 0.7,
    "centre_x": 1.5,
    "slope": 0.9,
}

#: Pass budgets the fallback rate is reported at. 2 and 7 are the two meshes'
#: measured certifying budgets at the fiducial (phase 0); 4 and 6 are the
#: intermediate production candidates.
DEFAULT_BUDGETS = (2, 4, 6, 7)


# ---------------------------------------------------------------------------
# A draw
# ---------------------------------------------------------------------------


@dataclass
class Draw:
    """One model in the draw set: offsets to the fiducial mass, plus provenance."""

    name: str
    kind: str  # "fiducial" | "walk" | "random"
    offsets: dict  # parameter -> physical offset from the fiducial value
    parameter: str | None = None  # walks only
    target: float | None = None  # walks only — where the bisection aimed
    achieved_d_log_l: float | None = None  # what it actually landed on
    n_evaluations: int = 0  # S0 evaluations the construction spent
    reached_target: bool | None = None
    mass_cls: str = "Isothermal"
    index: int | None = None  # random draws only
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "parameter": self.parameter,
            "target_d_log_l": self.target,
            "achieved_d_log_l": self.achieved_d_log_l,
            "reached_target": self.reached_target,
            "offsets": {k: float(v) for k, v in self.offsets.items()},
            "n_construction_evaluations": int(self.n_evaluations),
            "mass_cls": self.mass_cls,
            "index": self.index,
        }


# ---------------------------------------------------------------------------
# Mass construction (the only autolens-touching code here)
# ---------------------------------------------------------------------------


def fiducial_mass_values(instance) -> dict:
    """The fiducial mass parameters, read off the prior-median instance."""
    mass = instance.galaxies.lens.mass
    return {
        "centre_0": float(mass.centre[0]),
        "centre_1": float(mass.centre[1]),
        "ell_comps_0": float(mass.ell_comps[0]),
        "ell_comps_1": float(mass.ell_comps[1]),
        "einstein_radius": float(mass.einstein_radius),
        "slope": 2.0,
    }


def mass_from(base: dict, offsets: dict | None = None, *, mass_cls: str = "Isothermal"):
    """Build the draw's mass profile from the fiducial values plus ``offsets``.

    ``centre_x`` is an alias for ``centre_1`` — the *x* component of the
    ``(y, x)`` centre — so a walk reads as the parameter a modeller would name.

    ``mass_cls="PowerLaw"`` promotes the Isothermal to its ``slope`` generalisation.
    At ``slope = 2.0`` the two are the same profile, which is what makes the
    promotion checkable: the cell pins the promoted fiducial's S0 figure of merit
    against the Isothermal's before any slope row is measured.
    """
    import autolens as al

    offsets = dict(offsets or {})
    values = dict(base)
    for key, delta in offsets.items():
        target = "centre_1" if key == "centre_x" else ("centre_0" if key == "centre_y" else key)
        if target not in values:
            raise KeyError(f"mass_from: unknown mass parameter '{key}'")
        values[target] = values[target] + float(delta)

    centre = (values["centre_0"], values["centre_1"])
    ell_comps = (values["ell_comps_0"], values["ell_comps_1"])

    if mass_cls == "Isothermal":
        if abs(values["slope"] - 2.0) > 0.0:
            raise ValueError(
                "mass_from: an Isothermal has no slope; a slope offset needs mass_cls='PowerLaw'"
            )
        return al.mp.Isothermal(
            centre=centre, ell_comps=ell_comps, einstein_radius=values["einstein_radius"]
        )
    if mass_cls == "PowerLaw":
        return al.mp.PowerLaw(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=values["einstein_radius"],
            slope=values["slope"],
        )
    raise ValueError(f"mass_from: unknown mass_cls '{mass_cls}'")


def lens_galaxy_from(instance, mass, *, with_light: bool = True):
    """The cells' lens galaxy with ``mass`` swapped in (light and shear unchanged)."""
    import autolens as al

    lens = instance.galaxies.lens
    kwargs = dict(redshift=0.5, mass=mass, shear=lens.shear)
    if with_light:
        kwargs["bulge"] = lens.bulge
    return al.Galaxy(**kwargs)


# ---------------------------------------------------------------------------
# Random draws
# ---------------------------------------------------------------------------


def random_offsets(
    seed: int,
    n: int,
    *,
    sigma_scale: float = RANDOM_SIGMA_SCALE,
    prior_sigma: dict | None = None,
    parameters=RANDOM_PARAMETERS,
) -> list[dict]:
    """``n`` seeded Gaussian offset dicts at ``sigma_scale`` × the prior σ.

    ``numpy.random.default_rng(seed)`` with a fixed parameter order, so the same
    seed gives the same draws on every device and every mesh — that identity is
    what lets the A100 and CPU legs be compared row by row.
    """
    prior_sigma = PRIOR_SIGMA if prior_sigma is None else prior_sigma
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n)):
        # One vector per draw, drawn in a fixed parameter order: appending a
        # parameter later would then shift only the tail, not re-roll the set.
        draws.append({p: float(rng.normal(0.0, sigma_scale * prior_sigma[p])) for p in parameters})
    return draws


# ---------------------------------------------------------------------------
# The bisection
# ---------------------------------------------------------------------------


@dataclass
class BisectionResult:
    """Where a walk landed, and what it cost to get there."""

    delta: float
    achieved: float
    n_evaluations: int
    reached: bool
    history: list


def _evaluator_with_cache(evaluate, cache: dict):
    """Wrap ``evaluate(delta) -> Δlog L`` with a cache and an evaluation counter."""
    state = {"n": 0}

    def call(delta: float) -> float:
        key = round(float(delta), 12)
        if key not in cache:
            cache[key] = float(evaluate(key))
            state["n"] += 1
        return cache[key]

    return call, state


def bisect_to_target(
    evaluate,
    target: float,
    *,
    probe_delta: float,
    max_delta: float,
    sign: float = 1.0,
    max_evals: int = 12,
    rtol: float = 0.2,
    cache: dict | None = None,
) -> BisectionResult:
    """Scale a one-parameter offset until ``Δlog L`` lands on ``target``.

    ``evaluate(delta) -> Δlog L`` is the caller's S0 rebuild; ``target`` is
    negative. The routine assumes only that ``|Δlog L|`` grows with ``|delta|``,
    which is what a likelihood does as a parameter leaves its optimum — never
    that the relation has any particular form.

    The **first** evaluation seeds a quadratic model (``Δlog L ∝ −δ²``, exact for
    a locally quadratic likelihood), so the search usually starts inside the
    bracket instead of climbing a doubling ladder to it. Everything after that
    is a plain bisection on the bracket, and every evaluation — seed, bracket and
    bisection alike — goes through a **shared cache**, so the four targets of one
    walk cost far less than four independent searches.

    Stops when ``|achieved − target| <= rtol * |target|`` or when ``max_evals``
    *new* S0 evaluations have been spent (cached hits are free). ``reached``
    says which happened; a walk that did not reach its target is reported with
    the Δlog L it actually landed on and never re-labelled as the target.
    """
    if target >= 0.0:
        raise ValueError(f"bisect_to_target: target must be negative (got {target})")

    cache = {} if cache is None else cache
    call, state = _evaluator_with_cache(evaluate, cache)
    history: list = []

    def f(delta_abs: float) -> float:
        value = call(sign * abs(delta_abs))
        history.append({"delta": sign * abs(delta_abs), "d_log_l": value})
        return value

    def ok(value: float) -> bool:
        return abs(value - target) <= rtol * abs(target)

    # --- seed: one probe, then the quadratic prediction -------------------
    d_probe = abs(probe_delta)
    v_probe = f(d_probe)
    if not np.isfinite(v_probe):
        v_probe = -np.inf

    if ok(v_probe):
        return BisectionResult(sign * d_probe, v_probe, state["n"], True, history)

    if v_probe < 0.0 and np.isfinite(v_probe):
        guess = d_probe * math.sqrt(abs(target) / abs(v_probe))
    else:
        guess = d_probe
    guess = float(min(max(guess, 1e-9), max_delta))

    # --- bracket: lo is shallower than the target, hi is deeper -----------
    lo, v_lo = 0.0, 0.0
    hi, v_hi = None, None

    d = guess
    for _ in range(24):
        if state["n"] >= max_evals:
            break
        v = f(d)
        if ok(v):
            return BisectionResult(sign * d, v, state["n"], True, history)
        if not np.isfinite(v) or v < target:
            hi, v_hi = d, v
            break
        lo, v_lo = d, v
        if d >= max_delta:
            break
        d = min(d * 2.0, max_delta)

    if hi is None:
        # Never got deep enough than the target inside the budget/ceiling.
        best_d, best_v = (lo, v_lo) if lo > 0.0 else (d, v_probe)
        return BisectionResult(sign * best_d, best_v, state["n"], ok(best_v), history)

    # --- bisect ------------------------------------------------------------
    best_d, best_v = hi, v_hi
    while state["n"] < max_evals:
        mid = 0.5 * (lo + hi)
        if mid <= 0.0 or hi - lo <= 1e-12:
            break
        v = f(mid)
        if abs(v - target) < abs(best_v - target):
            best_d, best_v = mid, v
        if ok(v):
            return BisectionResult(sign * mid, v, state["n"], True, history)
        if not np.isfinite(v) or v < target:
            hi, v_hi = mid, v
        else:
            lo, v_lo = mid, v

    return BisectionResult(sign * best_d, best_v, state["n"], ok(best_v), history)


# ---------------------------------------------------------------------------
# The draw set
# ---------------------------------------------------------------------------


def walk_draws(
    evaluate_walk,
    *,
    parameters=WALK_PARAMETERS,
    targets=DEFAULT_TARGETS,
    max_evals: int = 12,
    rtol: float = 0.2,
) -> list[Draw]:
    """The one-parameter walks, one :class:`Draw` per (parameter, target).

    ``evaluate_walk(parameter, delta) -> Δlog L``. Targets are visited from the
    shallowest to the deepest so the shared per-parameter cache is warm by the
    time the expensive end of the walk is searched.
    """
    draws: list[Draw] = []
    for parameter in parameters:
        cache: dict = {}
        for target in sorted(targets, reverse=True):  # -10 first, -1e4 last
            result = bisect_to_target(
                lambda delta, p=parameter: evaluate_walk(p, delta),
                float(target),
                probe_delta=PROBE_DELTA[parameter],
                max_delta=MAX_DELTA[parameter],
                sign=WALK_SIGN[parameter],
                max_evals=max_evals,
                rtol=rtol,
                cache=cache,
            )
            draws.append(
                Draw(
                    name=f"walk_{parameter}_{_target_tag(target)}",
                    kind="walk",
                    parameter=parameter,
                    target=float(target),
                    offsets={parameter: result.delta},
                    achieved_d_log_l=result.achieved,
                    n_evaluations=result.n_evaluations,
                    reached_target=result.reached,
                    mass_cls="PowerLaw" if parameter == "slope" else "Isothermal",
                    history=result.history,
                )
            )
    return draws


def random_draws(seed: int, n: int, **kwargs) -> list[Draw]:
    """The seeded random family as :class:`Draw` objects."""
    return [
        Draw(
            name=f"random_{i:02d}",
            kind="random",
            offsets=offsets,
            index=i,
            mass_cls="Isothermal",
        )
        for i, offsets in enumerate(random_offsets(seed, n, **kwargs))
    ]


def _target_tag(target: float) -> str:
    magnitude = abs(float(target))
    if magnitude >= 1e4:
        return f"m{magnitude:.0e}".replace("+0", "").replace("+", "")
    return f"m{int(round(magnitude))}"


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------


def fallback_table(pass_counts, budgets=DEFAULT_BUDGETS) -> list[dict]:
    """Fraction of draws whose certification needs **more** than each budget.

    ``pass_counts`` entries are the pass at which each draw certified, or
    ``None`` for a draw that never certified inside the scan. A ``None`` counts
    as a fallback at **every** budget — that is precisely the production event
    the rate is measuring, and dropping it would report a rate over the draws
    that happened to be easy.
    """
    counts = list(pass_counts)
    n = len(counts)
    rows = []
    for budget in budgets:
        n_fallback = sum(1 for p in counts if p is None or int(p) > int(budget))
        rows.append(
            {
                "budget": int(budget),
                "n_draws": n,
                "n_fallback": int(n_fallback),
                "fallback_rate": (n_fallback / n) if n else None,
                "n_never_certified": sum(1 for p in counts if p is None),
            }
        )
    return rows


def rank_correlation(x, y) -> float | None:
    """Spearman ρ, ties averaged. ``None`` when either side is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # average ties
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(unique.size, dtype=float)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def quantile_summary(values, quantiles=(0.5, 0.9, 1.0)) -> dict:
    """Median / 90th / max of a list, ``None``-safe, for the random-draw rows."""
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"n": 0}
    array = np.asarray(finite, dtype=float)
    out = {"n": int(array.size), "min": float(array.min()), "mean": float(array.mean())}
    for q in quantiles:
        out[f"q{int(round(q * 100)):02d}"] = float(np.quantile(array, q))
    return out
