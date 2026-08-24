"""Detect the PositionsLH penalty's three characteristic non-smooth features.

Phase 4 Stage 1 (issue #159, ``results/notes/inference/PROGRAMME.md``
[H4.2]: "the hinge/argmax kinks and zero-gradient interior create *new*
failure modes ... measured, not assumed"). ``al.PositionsLH`` adds
``factor * max(max_sep - threshold, 0)`` to the negative log-likelihood
(autolens/analysis/positions.py), which is:

- a **C0 hinge** at ``max_sep == threshold`` — continuous value, jump
  discontinuity in the gradient (zero on the interior side, ``-factor *
  d(max_sep)/dtheta`` on the exterior side);
- an exact **zero-gradient interior plateau** everywhere ``max_sep <
  threshold`` — a gradient-search lane deep inside the fence gets no signal
  from the penalty at all;
- an **argmax-switch kink**: ``max_sep`` is
  ``max_i(furthest_distance_to_other_points(i))`` over the observed
  positions — a max of several smooth functions — so which pair of
  positions realises the maximum can change discontinuously as the mass
  model moves, producing a gradient kink in ``max_sep`` (and therefore in
  the penalty, whenever that region is also over threshold) independent of
  the threshold crossing itself.

Cheap reproducer: a synthetic single-plane Isothermal-mass tracer (no
imaging dataset — this is point-tracing only, sub-second) using the real
truth quad-image positions from ``dataset/imaging/hst/positions.json``
(and the truth mass/shear from ``dataset/imaging/hst/tracer.json`` as the
sweep anchor), swept over ``einstein_radius`` with ``jax.value_and_grad``
through the real ``autolens.analysis.positions.PositionsLH.
log_likelihood_penalty_from`` — the exact library code path, not a
reimplementation of its formula.
"""

from __future__ import annotations

import numpy as np

from hazards._anchor import maybe_anchor_from_pattern
from hazards._measure import Measurement, reachability_measurement
from hazards._record import Finding
from hazards.checks._base import HazardCheck, ScanContext

# Sweep is cheap (point-tracing only) — 401 points resolves the hinge/plateau
# without needing Transect B's 1e-4 fine-grid machinery.
_N_GRID = 401
_THETA_E_LO, _THETA_E_HI = 0.0, 3.0
_FACTOR = 1.0e8
_THRESHOLD = 0.3


def _truth_inputs(repo_root):
    import autolens as al

    dataset_path = repo_root / "dataset" / "imaging" / "hst"
    positions = al.from_json(file_path=dataset_path / "positions.json")
    tracer = al.from_json(file_path=dataset_path / "tracer.json")
    lens = min(tracer.galaxies, key=lambda g: float(g.redshift))
    return (
        positions,
        (float(lens.mass.ell_comps[0]), float(lens.mass.ell_comps[1])),
        (float(lens.shear.gamma_1), float(lens.shear.gamma_2)),
    )


class _FixedShapeAnalysis:
    """Duck-typed stand-in for an ``AnalysisLens``: only needs
    ``tracer_via_instance_from`` (what ``PositionsLH.log_likelihood_penalty_from``
    calls). ``instance`` here is the swept ``einstein_radius`` scalar itself —
    no ``af.ModelInstance`` / dataset / model needed for this cheap reproducer.
    """

    def __init__(self, ell_comps, shear_comps):
        self._ell_comps = ell_comps
        self._shear_comps = shear_comps

    def tracer_via_instance_from(self, instance):
        import autolens as al

        einstein_radius = instance
        mass = al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=self._ell_comps, einstein_radius=einstein_radius
        )
        shear = al.mp.ExternalShear(
            gamma_1=self._shear_comps[0], gamma_2=self._shear_comps[1]
        )
        lens = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
        source = al.Galaxy(redshift=1.0)
        return al.Tracer(galaxies=[lens, source])


def _penalty_value_and_grad_fn(positions, ell_comps, shear_comps, threshold: float, factor: float):
    """A jitted, vmap'd ``theta_e array -> (value, grad)`` over the REAL
    ``PositionsLH.log_likelihood_penalty_from`` — shared by the coarse sweep
    and the tight local hinge probe below, so both read off the identical
    code path."""
    import autolens as al
    import jax
    import jax.numpy as jnp

    analysis = _FixedShapeAnalysis(ell_comps, shear_comps)
    positions_lh = al.PositionsLH(
        positions=positions, threshold=threshold, log_likelihood_penalty_factor=factor
    )

    def penalty(theta_e):
        return positions_lh.log_likelihood_penalty_from(
            instance=theta_e, analysis=analysis, xp=jnp
        )

    return jax.jit(jax.vmap(jax.value_and_grad(penalty))), analysis


def _sweep(positions, ell_comps, shear_comps, threshold: float, factor: float):
    import autolens as al
    import jax.numpy as jnp

    value_and_grad_fn, analysis = _penalty_value_and_grad_fn(
        positions, ell_comps, shear_comps, threshold, factor
    )
    grid = jnp.linspace(_THETA_E_LO, _THETA_E_HI, _N_GRID)
    value, grad = value_and_grad_fn(grid)
    grid_np = np.asarray(grid)
    value_np = np.asarray(value)
    grad_np = np.asarray(grad)

    # Arm-independent geometry (max_sep + which pair attains it) at every
    # grid point — plain NumPy, no grad needed.
    max_sep = np.empty(_N_GRID)
    argmax_pairs = []
    for i, theta_e in enumerate(grid_np):
        tracer = analysis.tracer_via_instance_from(instance=float(theta_e))
        fit = al.SourceMaxSeparation(
            data=positions, noise_map=None, tracer=tracer, plane_redshift=None
        )
        seps = np.asarray(fit.furthest_separations_of_plane_positions.array)
        max_sep[i] = float(np.max(seps))
        argmax_pairs.append(
            frozenset(np.flatnonzero(np.isclose(seps, max_sep[i], rtol=1e-9, atol=1e-9)).tolist())
        )

    return grid_np, value_np, grad_np, max_sep, argmax_pairs


def _local_hinge_probe(positions, ell_comps, shear_comps, threshold: float, factor: float, x_cross: float):
    """Tight ``x_cross +/- 1e-6`` probe of the REAL penalty function.

    The coarse sweep's grid spacing (~0.0075 for the default 401-pt grid)
    is far too wide to read the VALUE gap at a crossing directly — with
    factor=1e8 a genuinely continuous but steep branch can differ by O(1e6)
    between grid-adjacent points, which would misreport as a value
    discontinuity. This tight local probe is the actual C0-continuity
    evidence; the coarse grid's ``grad`` jump (an exact one-sided derivative
    at each concrete point, not a finite difference) is unaffected by this
    and needs no such correction.
    """
    import jax.numpy as jnp

    eps = 1.0e-6
    value_and_grad_fn, _ = _penalty_value_and_grad_fn(positions, ell_comps, shear_comps, threshold, factor)
    xs = jnp.asarray([max(x_cross - eps, _THETA_E_LO), min(x_cross + eps, _THETA_E_HI)])
    value, grad = value_and_grad_fn(xs)
    return {
        "eps": eps,
        "x_cross": x_cross,
        "penalty_left": float(value[0]),
        "penalty_right": float(value[1]),
        "value_continuity_gap": float(abs(value[1] - value[0])),
        "dpenalty_left": float(grad[0]),
        "dpenalty_right": float(grad[1]),
        "gradient_jump": float(grad[1] - grad[0]),
    }


class PositionsPenaltyCheck(HazardCheck):
    name = "positions_penalty"
    subject = "likelihood"

    def run(self, context: ScanContext) -> list[Finding]:
        positions, ell_comps, shear_comps = _truth_inputs(context.repo_root)
        grid, value, grad, max_sep, argmax_pairs = _sweep(
            positions, ell_comps, shear_comps, _THRESHOLD, _FACTOR
        )

        penalty_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoLens",
            path="autolens/analysis/positions.py",
            pattern="penalty = self.log_likelihood_penalty_factor * (max_separation - self.threshold)",
            symbol="autolens.analysis.positions.PositionsLH.log_likelihood_penalty_from",
        )
        max_sep_anchor = maybe_anchor_from_pattern(
            context.workspace_root,
            repo="PyAutoLens",
            path="autolens/point/max_separation.py",
            pattern="def furthest_separations_of_plane_positions",
            after=15,
            symbol=(
                "autolens.point.max_separation.SourceMaxSeparation."
                "furthest_separations_of_plane_positions"
            ),
        )
        reachability = reachability_measurement(
            reachable_via=["AnalysisDataset.positions_likelihood_list -> PositionsLH.log_likelihood_penalty_from"]
        )

        findings: list[Finding] = []

        # --- threshold-hinge -------------------------------------------------
        crossing_indices = []
        diff = max_sep - _THRESHOLD
        for i in range(1, len(grid)):
            if (diff[i - 1] < 0.0) != (diff[i] < 0.0):
                crossing_indices.append(i)
        # Interpolate the crossing theta_e from the coarse grid, then probe
        # it TIGHTLY (+/- 1e-6) — the coarse grid's own spacing is far too
        # wide to read the value gap directly at factor=1e8 (see
        # _local_hinge_probe's docstring).
        hinge_probes = []
        for i in crossing_indices:
            frac = -diff[i - 1] / (diff[i] - diff[i - 1]) if diff[i] != diff[i - 1] else 0.0
            x_cross = float(grid[i - 1] + frac * (grid[i] - grid[i - 1]))
            hinge_probes.append(
                _local_hinge_probe(positions, ell_comps, shear_comps, _THRESHOLD, _FACTOR, x_cross)
            )
        hazard_persists = len(hinge_probes) > 0 and max(
            abs(p["gradient_jump"]) for p in hinge_probes
        ) > 1e-6
        if hazard_persists:
            max_jump = max(abs(p["gradient_jump"]) for p in hinge_probes)
            max_value_gap = max(p["value_continuity_gap"] for p in hinge_probes)
            findings.append(
                Finding(
                    finding_id="likelihood.positions-penalty.threshold-hinge",
                    title="PositionsLH penalty has a C0 hinge at the threshold crossing",
                    summary=(
                        f"Sweeping einstein_radius in [{_THETA_E_LO}, {_THETA_E_HI}] ({_N_GRID} pts, "
                        f"threshold={_THRESHOLD}, factor={_FACTOR:g}), a tight (+/-1e-6) probe at "
                        f"each of {len(hinge_probes)} max_sep==threshold crossing(s) shows the "
                        f"penalty VALUE is continuous there (max gap {max_value_gap:.3e}) while its "
                        f"jax-autodiff GRADIENT jumps by up to {max_jump:.3e} — zero on the interior "
                        "side, nonzero on the exterior side. A gradient optimizer sees a step-"
                        "function force switching on at the fence, not a smooth potential."
                    ),
                    hazard_class="nonsmooth_objective",
                    tier=2,
                    subject="likelihood",
                    subject_name="positions_penalty",
                    backends=("jax",),
                    measurements=(
                        Measurement(
                            basis="error_curve",
                            value=max_jump,
                            unit="d_penalty_d_einstein_radius_jump",
                            details={
                                "n_crossings": len(hinge_probes),
                                "hinge_probes": hinge_probes,
                                "threshold": _THRESHOLD,
                                "factor": _FACTOR,
                            },
                        ),
                        reachability,
                    ),
                    anchors=tuple(a for a in (penalty_anchor,) if a is not None),
                    code_exists=True,
                    reachable_via=(
                        "AnalysisDataset.positions_likelihood_list -> "
                        "PositionsLH.log_likelihood_penalty_from",
                    ),
                    blocked_by=(),
                    affects_science=None,
                    backend_reachability={"jax": {"gradient_at_crossing": "discontinuous"}},
                    reproducer={
                        "theta_e_grid": grid.tolist(),
                        "penalty": value.tolist(),
                        "dpenalty": grad.tolist(),
                        "max_sep": max_sep.tolist(),
                        "recommendation": (
                            "Documented, not a defect: a hard-threshold penalty is a fence by "
                            "design (see #159 Stage 1 RESULTS.md for the full transect). A "
                            "gradient MAP search initialised inside the fence sees zero signal "
                            "from positions until a lane actually reaches the boundary."
                        ),
                    },
                )
            )

        # --- interior plateau --------------------------------------------------
        interior_mask = max_sep < _THRESHOLD
        interior_grad_max_abs = float(np.max(np.abs(grad[interior_mask]))) if interior_mask.any() else None
        interior_value_max_abs = (
            float(np.max(np.abs(value[interior_mask]))) if interior_mask.any() else None
        )
        is_exact_zero = interior_mask.any() and interior_grad_max_abs == 0.0 and interior_value_max_abs == 0.0
        if interior_mask.any():
            findings.append(
                Finding(
                    finding_id="likelihood.positions-penalty.interior-plateau",
                    title="PositionsLH penalty is an exact-zero, zero-gradient plateau inside threshold",
                    summary=(
                        f"Across {int(interior_mask.sum())}/{_N_GRID} grid points with "
                        f"max_sep < threshold, penalty value and gradient are exactly zero "
                        f"(max |value|={interior_value_max_abs:.3e}, max |grad|="
                        f"{interior_grad_max_abs:.3e}) — a gradient-search lane deep inside the "
                        "fence gets NO signal at all from the positions term, by construction."
                    ),
                    hazard_class="zero_gradient_region",
                    tier=1,
                    subject="likelihood",
                    subject_name="positions_penalty",
                    backends=("jax",),
                    measurements=(
                        Measurement(
                            basis="error_curve",
                            value=interior_grad_max_abs,
                            unit="max_abs_dpenalty_on_interior",
                            details={
                                "n_interior_points": int(interior_mask.sum()),
                                "n_total_points": _N_GRID,
                                "max_abs_value_on_interior": interior_value_max_abs,
                                "is_exact_zero_plateau": bool(is_exact_zero),
                            },
                        ),
                        reachability,
                    ),
                    anchors=tuple(a for a in (penalty_anchor,) if a is not None),
                    code_exists=True,
                    reachable_via=(
                        "AnalysisDataset.positions_likelihood_list -> "
                        "PositionsLH.log_likelihood_penalty_from",
                    ),
                    blocked_by=(),
                    affects_science=None,
                    backend_reachability={"jax": {"interior_gradient": "exact_zero"}},
                    reproducer={
                        "theta_e_grid": grid.tolist(),
                        "penalty": value.tolist(),
                        "dpenalty": grad.tolist(),
                        "recommendation": (
                            "Expected behaviour, not a bug: positions is a hard-fence prior, not "
                            "a smooth potential well. Relevant to [H4.2] — a MultiStart lane that "
                            "starts and stays inside the fence gets zero navigational signal from "
                            "this term; whether that matters depends on the base likelihood's own "
                            "landscape there (untested at Stage 1 — Stage 2's remit)."
                        ),
                    },
                )
            )

        # --- argmax-switch kink --------------------------------------------
        switches = []
        for i in range(1, len(grid)):
            if argmax_pairs[i] != argmax_pairs[i - 1]:
                switches.append(
                    {
                        "theta_e_before": float(grid[i - 1]),
                        "theta_e_after": float(grid[i]),
                        "pair_before": sorted(argmax_pairs[i - 1]),
                        "pair_after": sorted(argmax_pairs[i]),
                        "over_threshold": bool(max_sep[i] > _THRESHOLD),
                    }
                )
        if switches:
            n_over_threshold = sum(1 for s in switches if s["over_threshold"])
            findings.append(
                Finding(
                    finding_id="likelihood.positions-penalty.argmax-switch",
                    title="Which position-pair sets max_sep switches discontinuously across the sweep",
                    summary=(
                        f"{len(switches)} argmax-switch location(s) found in "
                        f"[{_THETA_E_LO}, {_THETA_E_HI}] ({len(positions)} truth positions, "
                        f"{n_over_threshold} of them where max_sep is already over threshold — "
                        "i.e. a second, threshold-independent gradient kink in the ACTIVE penalty "
                        "region, from max_sep itself being a max of several smooth per-position "
                        "functions)."
                    ),
                    hazard_class="nonsmooth_objective",
                    tier=2,
                    subject="likelihood",
                    subject_name="positions_penalty",
                    backends=("jax",),
                    measurements=(
                        Measurement(
                            basis="error_curve",
                            value=float(len(switches)),
                            unit="n_argmax_switches",
                            details={"switches": switches},
                        ),
                        reachability,
                    ),
                    anchors=tuple(a for a in (max_sep_anchor,) if a is not None),
                    code_exists=True,
                    reachable_via=(
                        "SourceMaxSeparation.furthest_separations_of_plane_positions -> "
                        "PositionsLH.log_likelihood_penalty_from",
                    ),
                    blocked_by=(),
                    affects_science=None,
                    backend_reachability={"jax": {"argmax_pair": "can_switch_discontinuously"}},
                    reproducer={
                        "theta_e_grid": grid.tolist(),
                        "max_sep": max_sep.tolist(),
                        "n_positions": len(positions),
                        "recommendation": (
                            "Only matters where over_threshold=true (the penalty is active there); "
                            "an argmax-switch on the interior is invisible (both sides plateau at "
                            "zero). See #159 Stage 1 Transect A/B for the full-likelihood "
                            "(imaging/mge/hst) measurement of where these land relative to the "
                            "threshold crossings."
                        ),
                    },
                )
            )

        return findings
