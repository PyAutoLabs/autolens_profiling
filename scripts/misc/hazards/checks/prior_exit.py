"""Detect the prior-exit failure mode: a finite likelihood made non-finite by the prior.

The objective a MAP search minimises is ``-2 * (log_likelihood + sum(log_prior))``,
not the likelihood alone. A ``UniformPrior`` returns ``-inf`` outside its box, so
a parameter vector whose *likelihood* is perfectly finite has a non-finite
objective the instant any one coordinate leaves its prior box.

That matters because ``MultiStartGradient`` steps in PHYSICAL parameter space
with nothing holding it inside the box. A lane that oversteps by a fraction of a
box width reads as non-finite, is counted dead, and — with ``resurrect=False`` —
is never redrawn, while continuing to step and to consume a full
likelihood-and-gradient evaluation whose output is discarded.

This is a *search* hazard rather than a numerical one: no value is wrong, the
population is silently truncated. On the reference `imaging/mge` cell it removed
14 of 16 lanes, and the likelihood never went non-finite once in ~7200
lane-steps across three arms (autolens_profiling#128).

It is blocked, not fixed, by ``ClipperPriorBox`` (PyAutoFit#1477), which projects
each step back onto prior support. The ``blocked_by`` anchor records that
mitigation, so the entry stays honest about the hazard still existing in the
default path.

Since PyAutoFit#1489 (2026-08-18) the NumPy path enforces the same bound as the
JAX path — 0.0 inside the box, ``-inf`` outside, through the same ``xp.where``.
Before that fix the NumPy path short-circuited to a scalar 0.0 without
evaluating the bound at all, so the hazard was reachable only under JAX; the
check measured that asymmetry, and this module's first CI failure was that
measurement correctly detecting the flip. The hazard itself (a lane stepping
out of support goes non-finite and dies) is unchanged — it is now simply
reachable on both backends.
"""

from __future__ import annotations

import numpy as np

from hazards._anchor import maybe_anchor_from_pattern
from hazards._measure import prior_mass_measurement, reachability_measurement
from hazards._record import Finding
from hazards.checks._base import HazardCheck, ScanContext

# How far outside the box the Monte Carlo band reaches, as a fraction of the box
# width. 3% is the scale #128 actually observed: lanes died having overstepped by
# roughly that much, so the band is sized to the real failure rather than to a
# number chosen to make the hazard look large.
_OVERSTEP_FRACTION = 0.03


class PriorExitCheck(HazardCheck):
    """A finite likelihood driven non-finite by leaving a bounded prior's support."""

    name = "prior_exit"
    subject = "likelihood"

    def run(self, context: ScanContext) -> list[Finding]:
        import autofit as af

        anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoFit",
            path="autofit/mapper/prior/uniform.py",
            pattern="return xp.where(in_bounds, xp.zeros_like(value), -xp.inf)",
            symbol="autofit.mapper.prior.uniform.UniformPrior.log_prior_from_value",
        )

        # The mitigation, recorded as what BLOCKS the hazard rather than as what
        # removes it: the clipper is opt-in, so the default search path still
        # reaches the site.
        blocked_by = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoFit",
            path="autofit/non_linear/clipper.py",
            pattern="class ClipperPriorBox",
            symbol="autofit.non_linear.clipper.ClipperPriorBox",
        )

        lower, upper = 0.0, 1.0
        prior = af.UniformPrior(lower_limit=lower, upper_limit=upper)

        # Draw across a band that straddles the boundary, so the measurement is
        # "how much of a neighbourhood a stepping search can reach is finite",
        # not the trivially-1.0 answer you get by sampling inside the box only.
        margin = _OVERSTEP_FRACTION * (upper - lower)
        rng = np.random.default_rng(context.seed)
        values = rng.uniform(lower - margin, upper + margin, size=context.sample_count)
        outside = (values < lower) | (values > upper)

        # Since PyAutoFit#1489 both backends enforce the bound through the same
        # `xp.where` — 0.0 inside, -inf outside — so a NumPy scan measures the
        # same support boundary the JAX-native gradient searches hit. The parity
        # measurement is kept (rather than assumed) so the record stays honest
        # if the pre-#1489 asymmetry — a NumPy path that short-circuited to a
        # scalar 0.0 without evaluating the bound — ever returns; in that world
        # only a JAX scan can adjudicate the hazard.
        numpy_log_prior = np.asarray(prior.log_prior_from_value(value=values, xp=np), dtype=float)
        numpy_finite = np.isfinite(numpy_log_prior)
        numpy_enforces_bounds = bool(np.any(~numpy_finite))

        jax_available = "jax" in context.backends
        if jax_available:
            import jax.numpy as jnp

            log_prior = np.asarray(
                prior.log_prior_from_value(value=jnp.asarray(values), xp=jnp), dtype=float
            )
            finite = np.isfinite(log_prior)
        else:
            finite = numpy_finite

        # The JAX path (the one the gradient searches run) always adjudicates;
        # a NumPy-only scan adjudicates exactly when the NumPy path evaluates
        # the bound. An unadjudicated scan must not report the hazard resolved.
        adjudicated = jax_available or numpy_enforces_bounds
        hazard_persists = bool(np.any(~finite)) if adjudicated else True

        measurements = (
            prior_mass_measurement(
                finite,
                sample_count=context.sample_count,
                seed=context.seed,
                prior={
                    "distribution": "uniform",
                    "lower_limit": lower,
                    "upper_limit": upper,
                    "sampled_band": [lower - margin, upper + margin],
                    "overstep_fraction": _OVERSTEP_FRACTION,
                },
            ),
            reachability_measurement(
                reachable_via=[
                    # The search steps in physical space, so nothing constrains
                    # a step to stay inside the box.
                    "multi_start_gradient_physical_stepping",
                    "bounded_prior_objective",
                ],
                blocked_paths=(
                    {"clipper_prior_box": "projects each step back onto prior support"}
                    if blocked_by is not None
                    else {}
                ),
            ),
        )

        finding = Finding(
            finding_id="likelihood.prior_support.prior-exit-non-finite",
            title="Bounded prior makes a finite likelihood non-finite outside its box",
            summary=(
                "The MAP objective is -2*(log_likelihood + sum(log_prior)); a "
                "UniformPrior returns -inf outside its box, so a lane that steps "
                "out of prior support goes non-finite even though its likelihood "
                "is finite. MultiStartGradient steps in physical space with no "
                "projection, and with resurrect=False a lane that oversteps is "
                "counted dead for every remaining step while still paying full "
                "likelihood-and-gradient cost. Blocked by ClipperPriorBox, which "
                "is opt-in, so the default path still reaches it."
            ),
            hazard_class="prior-exit-non-finite",
            tier=1,
            subject="likelihood",
            subject_name="prior_support",
            backends=tuple(context.backends),
            measurements=measurements,
            anchors=tuple(a for a in (anchor,) if a is not None),
            code_exists=anchor is not None,
            reachable_via=(
                "multi_start_gradient_physical_stepping",
                "bounded_prior_objective",
            ),
            blocked_by=tuple(a for a in (blocked_by,) if a is not None),
            affects_science=True,
            backend_reachability={
                # Symmetric since PyAutoFit#1489: both paths return -inf outside
                # the box, so the lane death is reachable on either backend. The
                # measured branch below keeps the record honest if the NumPy
                # path ever regresses to the pre-#1489 unevaluated-bound
                # behaviour — recorded as observed, not as safe.
                "numpy": {
                    "log_prior_inside_box": "zero",
                    "log_prior_outside_box": (
                        "non_finite (since PyAutoFit#1489)"
                        if numpy_enforces_bounds
                        else "zero (bound not evaluated on this path)"
                    ),
                    "likelihood_outside_box": "unaffected",
                },
                **(
                    {
                        "jax": {
                            "log_prior_inside_box": "finite",
                            "log_prior_outside_box": "non_finite",
                            "likelihood_outside_box": "unaffected",
                        }
                    }
                    if jax_available
                    else {}
                ),
            },
            reproducer={
                "hazard_persists": hazard_persists,
                "adjudicated": adjudicated,
                "sampled": int(context.sample_count),
                "outside_box": int(np.count_nonzero(outside)),
                "non_finite_log_prior": (int(np.count_nonzero(~finite)) if adjudicated else None),
                # Every non-finite point is an out-of-box point and vice versa:
                # the -inf tracks the box edge exactly, which is what makes this
                # a support boundary rather than a numerical instability.
                "non_finite_iff_outside": (
                    bool(np.array_equal(~finite, outside)) if adjudicated else None
                ),
                "numpy_non_finite_outside_box": int(np.count_nonzero(~numpy_finite)),
                "numpy_evaluates_bound": numpy_enforces_bounds,
                "mitigation": "af.ClipperPriorBox() on the search",
                "observed_on": (
                    "autolens_profiling#128 imaging/mge hst: 14 of 16 lanes lost, "
                    "1446/2400 value-NaN lane-steps, likelihood finite throughout"
                ),
            },
        )

        return [finding]
