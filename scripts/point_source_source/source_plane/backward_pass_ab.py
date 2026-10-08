"""
JAX Profiling: Source-Plane Point-Source Backward-Pass A/B (phase 2b)
=====================================================================

An interleaved, in-process A/B of *how the gradient* of the production
source-plane point-source likelihood is computed (autolens_profiling #325,
single-source only). Phase 2a (``pytree_input_ab.py``) put the RAL CPU
``jax.value_and_grad`` call at 2.26x the fused forward (+0.185 ms) and promoted
the backward pass as the largest absolute residue.

Why the backward pass is expensive: the likelihood reads the lensing Hessian
through ``LensCalc.hessian_from`` -> ``LensCalc._hessian_via_jax``, a per-position
``jax.jacfwd`` of ``deflections_yx_scalar`` under ``jnp.vectorize``, so
``jax.value_and_grad`` runs reverse mode over a forward-mode Hessian for only
5 (solved) / 7 (plain) model parameters on 4 positions. Every lever below is
prototyped INSIDE this cell; no library is edited.

Routes (per lane; every route returns ``(log L, gradient pytree)``)
-------------------------------------------------------------------

- ``rev`` — production control: ``jax.value_and_grad(ll)``, Hessian by
  ``jax.jacfwd`` (the unpatched library method).
- ``fwd`` — forward-mode gradient: ``jax.jacfwd(ll, has_aux=True)`` with the
  value returned as aux. The argument is the same registered ``ModelInstance``
  pytree as ``rev`` (its leaves ARE the flat 5/7-parameter vector; ``jacfwd``
  pushes one tangent per scalar leaf), so the A/B isolates the AD mode from the
  argument-handling cost phase 2a already measured.
- ``rev_jacrev`` — ``rev`` with the Hessian built by ``jax.jacrev`` instead of
  ``jax.jacfwd`` over ``deflections_yx_scalar``.
- ``rev_analytic`` — ``rev`` with the SIE Hessian in closed form from
  ``Isothermal.convergence_2d_from`` / ``shear_yx_2d_from``:
  ``H_xx = kappa + gamma_1``, ``H_yy = kappa - gamma_1``, ``H_xy = H_yx = gamma_2``,
  where ``shear_yx_2d_from`` returns ``[:, 0] = gamma_2`` and ``[:, 1] = gamma_1``
  (autogalaxy's ``gamma_1 = (H_xx - H_yy) / 2``, ``gamma_2 = H_xy``). The mapping
  is verified against the ``jacfwd`` Hessian by the correctness gate, not assumed.
  The analytic Hessian REFUSES (raises at trace time) unless the tracer is two
  planes with exactly one mass profile of type ``Isothermal``.
- ``fwd_analytic`` — ``fwd`` with the analytic Hessian.

A secondary ``forward`` row times the plain likelihood under the three Hessian
builders (``jacfwd`` / ``jacrev`` / ``analytic``) for context: it says how much of
any saving is forward-pass and how much is backward-pass.

Hessian swaps
-------------

``LensCalc._hessian_via_jax`` is replaced by a scoped monkeypatch
(``hessian_patch``, a context manager) that restores the original on exit and
asserts the restore. Each route is traced, lowered and compiled INSIDE its patch
scope from a FRESH closure (fresh ``AnalysisPoint``) after ``jax.clear_caches()``;
the timing loop then calls those ``Compiled`` executables, which captured the
patched function at trace time and never retrace. Every patched Hessian counts
its trace-time invocations (asserted > 0), and every route's lowered StableHLO
text is hashed: the hashes must be pairwise distinct within a row, so no route
can have been served another's trace.

Correctness gate (before any timing; a failing route is reported, never timed)
------------------------------------------------------------------------------

1. Hessian components, analytic and jacrev vs jacfwd, rtol 1e-10 (error scaled
   by the largest |component| at each point) at the dataset positions, at a
   near-critical set (each dataset position moved along its ray from the lens
   centre to the tangential critical curve, found by bisection on
   ``1 - kappa - |gamma|``, then stepped off it by relative 1e-2 / 1e-4 / 1e-6
   either side) and for perturbed models (prior medians + ``PRNGKey`` 0..15
   draws).
2. Every route's log L equals ``rev`` to rtol 1e-10 and its gradient equals
   ``rev``'s to rtol 1e-8 (atol 1e-12 x max|grad|), finite AND non-zero, over the
   prior medians and ``PRNGKey`` 0..15 draws (``U(0.25, 0.75)`` unit cube ->
   ``vector_from_unit_vector``). ``autofit.jax.register_model(model)`` is
   load-bearing: without it ``jax.grad`` of an ``AnalysisPoint`` likelihood is
   silently all-zero.
3. Eager (``jax.disable_jit()``) == JIT per route, same tolerances, on the prior
   medians and ``PRNGKey`` 0 and 1.

Timing
------

Protocol from ``pytree_input_ab.py``: lower / compile / first-call seconds per
route; >= 3 warm calls; ``--rounds`` round-robin rounds with the route order
rotated each round and ``--calls`` calls per route per round, walking a
fixed-seed 16-instance stream (entry 0 the prior medians, the rest
``vector_from_unit_vector(U(0.25, 0.75))``); every call ``block_until_ready``'d.
Per route: median / p10 / p90 / min / max ms; median ratio vs ``rev`` with a
bootstrap 90 % CI; median ms saved vs ``rev`` with a bootstrap 90 % CI; XLA
flops where available.

Pre-registered verdict (issue #325; the RAL CPU row decides)
------------------------------------------------------------

A route is **GO** for a library-first phase 2c iff, on the host, its
``value_and_grad``-equivalent call saves >= 0.05 ms AND >= 15 % vs ``rev`` with
the 90 % CIs excluding the bar (saved-ms CI low >= 0.05 and ratio CI high
<= 0.85) and its correctness gate is green. Compile time is reported but is not
part of the rule. The JSON evaluates it for every config; only
``hpc_ral_cpu_fp64`` decides.

Since #362 fix phase 3 the rule is the shared
``likelihood_breakdown.ab_verdict.ab_rule_verdict`` (bars unchanged): a red
correctness gate is NO_GO; GO when both intervals clear their bars; NO_GO only
when an interval is wholly on the wrong side; INCONCLUSIVE otherwise or below
``MIN_AB_ROUNDS`` rounds — an overlapping CI is no longer written as no-go.

Output
------

``results/breakdown/point_source_source/backward_pass_ab_<config_name>.{json,png}``.
The JSON carries no top-level ``autolens_version`` (versions live under
``library_versions``): a note-backed A/B, not a README auto-table row.
"""

import os
import sys
import time
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_MISC = str(_ROOT / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import contextlib
import hashlib
import json

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib
import numpy as np

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from likelihood_breakdown.ab_verdict import (  # noqa: E402
    AB_CONFIDENCE,
    AT_LEAST,
    AT_MOST,
    MIN_AB_ROUNDS,
    NO_GO,
    Criterion,
    ab_rule_verdict,
)
from likelihood_breakdown.provenance import source_revisions, thread_environment  # noqa: E402
from likelihood_breakdown.round_bootstrap import (  # noqa: E402
    round_median_ratio,
    round_median_saving,
)
from likelihood_breakdown.timing import block  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    machine_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

_WALL_START = time.perf_counter()
_LOADAVG_START = os.getloadavg()

_cli = parse_profile_cli()
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--rounds", type=int, default=None)
_cell_parser.add_argument("--calls", type=int, default=None)
_cell_parser.add_argument("--n-stream", type=int, default=16)
_cell_parser.add_argument("--warm", type=int, default=3)
_cell_parser.add_argument("--seed", type=int, default=325)
_cell_parser.add_argument("--gate-keys", type=int, default=16, help="PRNGKey 0..N-1 draws")
_cell_parser.add_argument("--quick", action="store_true", help="rounds=5, calls=5")
_args = _cli.parse_cell_args(_cell_parser)

QUICK = bool(_args.quick)
N_ROUNDS = _args.rounds if _args.rounds is not None else (5 if QUICK else 20)
N_CALLS = _args.calls if _args.calls is not None else (5 if QUICK else 20)
N_STREAM = _args.n_stream
N_WARM = max(3, _args.warm)
SEED = _args.seed
N_GATE_KEYS = _args.gate_keys
if N_ROUNDS < 1 or N_CALLS < 1 or N_STREAM < 1 or N_GATE_KEYS < 1:
    raise SystemExit("--rounds, --calls, --n-stream and --gate-keys must be positive")

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autogalaxy as ag  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402

INSTRUMENT = "simple"
LANES = ("solved", "plain")
#: route -> (AD mode, Hessian builder)
GRAD_ROUTES = {
    "rev": ("rev", "jacfwd"),
    "fwd": ("fwd", "jacfwd"),
    "rev_jacrev": ("rev", "jacrev"),
    "rev_analytic": ("rev", "analytic"),
    "fwd_analytic": ("fwd", "analytic"),
}
#: forward-only context row: route -> Hessian builder
FORWARD_ROUTES = {"jacfwd": "jacfwd", "jacrev": "jacrev", "analytic": "analytic"}
CONTROL = {"grad": "rev", "forward": "jacfwd"}
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
HESSIAN_RTOL = 1.0e-10
VALUE_RTOL = 1.0e-10
GRAD_RTOL = 1.0e-8
GRAD_ATOL_SCALE = 1.0e-12
NEAR_CRITICAL_STEPS = (1.0e-2, 1.0e-4, 1.0e-6)
#: Phase-2c go/no-go thresholds (issue #325).
GO_MIN_SAVED_MS = 0.05
GO_MIN_FRACTION = 0.15

_THREADS_AFTER_FIRST_COMPILE: dict = {}


def _thread_count() -> int | None:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Model / analysis builders — copied from pytree_input_ab.py (flat scripts)
# ---------------------------------------------------------------------------


def _mass_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    return mass


def _model(*, solved: bool):
    lens = af.Model(al.Galaxy, redshift=0.5, mass=_mass_model())
    if solved:
        point = af.Model(al.ps.PointSolved)
    else:
        point = af.Model(al.ps.PointFlux)
        point.centre.centre_0 = af.GaussianPrior(mean=0.07, sigma=0.005)
        point.centre.centre_1 = af.GaussianPrior(mean=0.07, sigma=0.005)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


dataset_path = _ROOT / "dataset" / "point_source" / INSTRUMENT
auto_simulate_if_missing(
    dataset_path,
    dataset_type="point_source",
    instrument=INSTRUMENT,
    workspace_root=_ROOT,
)
dataset = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")


def _analysis(fit_positions_cls, *, use_jax: bool):
    return al.AnalysisPoint(
        dataset=dataset,
        solver=None,
        fit_positions_cls=fit_positions_cls,
        use_jax=use_jax,
    )


MODELS = {"solved": _model(solved=True), "plain": _model(solved=False)}
FIT_CLS = {"solved": al.FitPositionsSourceSolved, "plain": al.FitPositionsSource}
# Load-bearing: without registration the ModelInstance is not a pytree and
# jax.grad w.r.t. it is silently all-zero.
for _model_obj in MODELS.values():
    register_model_pytrees(_model_obj)


# ---------------------------------------------------------------------------
# Statistics (protocol from pytree_input_ab.py, + a bootstrap of the saving)
# ---------------------------------------------------------------------------


def _stats_ms(seconds) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1.0e3
    return {
        "n": int(ms.size),
        "median_ms": float(np.median(ms)),
        "p10_ms": float(np.percentile(ms, 10)),
        "p90_ms": float(np.percentile(ms, 90)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "mean_ms": float(ms.mean()),
    }


def _median_ratio(numerator, denominator, seed: int) -> dict:
    """median(numerator) / median(denominator), with a paired whole-round bootstrap 90 % interval.

    Resamples whole rounds with the same indices for both arms (#362 fix phase 4,
    the shared ``likelihood_breakdown.round_bootstrap``); ``effective_n`` is
    ``N_ROUNDS``. It replaced an iid, unpaired resampling of individual calls.
    """
    return round_median_ratio(
        numerator, denominator, n_rounds=N_ROUNDS, seed=seed, samples=BOOTSTRAP_SAMPLES
    )


def _median_saving_ms(control, route, seed: int) -> dict:
    """median(control) - median(route) in ms, with a paired whole-round bootstrap 90 % interval.

    Resamples whole rounds with the same indices for both arms (#362 fix phase 4,
    the shared ``likelihood_breakdown.round_bootstrap``); ``effective_n`` is
    ``N_ROUNDS``. It replaced an iid, unpaired resampling of individual calls.
    """
    return round_median_saving(
        control, route, n_rounds=N_ROUNDS, seed=seed, samples=BOOTSTRAP_SAMPLES
    )


def _flops(compiled):
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list | tuple):
            cost = cost[0] if cost else {}
        value = (cost or {}).get("flops")
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001 — cost analysis is optional on some backends
        return f"unavailable: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# The three Hessian builders and the scoped monkeypatch
# ---------------------------------------------------------------------------

_ORIGINAL_HESSIAN = ag.LensCalc.__dict__["_hessian_via_jax"]
_TRACE_COUNTS = {"jacfwd": 0, "jacrev": 0, "analytic": 0}


def _hessian_jacrev(self, grid, xp):
    """Verbatim ``LensCalc._hessian_via_jax`` with ``jax.jacrev`` for ``jax.jacfwd``."""
    _TRACE_COUNTS["jacrev"] += 1
    pixel_scales = getattr(grid, "pixel_scales", (0.05, 0.05))
    y = jnp.array(grid[:, 0])
    x = jnp.array(grid[:, 1])

    def _hessian_single(y_scalar, x_scalar):
        return jnp.stack(
            jax.jacrev(self.deflections_yx_scalar, argnums=(0, 1))(y_scalar, x_scalar, pixel_scales)
        )

    h = jnp.vectorize(_hessian_single, signature="(),()->(i,i)")(y, x)
    return (
        xp.array(h[..., 0, 0]),
        xp.array(h[..., 0, 1]),
        xp.array(h[..., 1, 0]),
        xp.array(h[..., 1, 1]),
    )


class AnalyticHessianRefused(RuntimeError):
    """The tracer is not a single two-plane SIE; the closed form does not apply."""


def _single_isothermal(lens_calc):
    """The one ``Isothermal`` behind a two-plane ``LensCalc``, or raise (never approximate)."""
    deflections = lens_calc.deflections_yx_2d_from
    tracer = getattr(deflections, "__self__", None)
    if tracer is None or getattr(deflections, "__name__", "") != "deflections_yx_2d_from":
        raise AnalyticHessianRefused(
            f"LensCalc deflections are {deflections!r}, not a two-plane "
            "Tracer.deflections_yx_2d_from (multi-plane or non-tracer source)"
        )
    planes = getattr(tracer, "planes", None)
    if planes is None or len(planes) != 2:
        raise AnalyticHessianRefused(f"tracer has {len(planes or [])} planes, need exactly 2")
    profiles = [
        profile
        for galaxy in tracer.galaxies
        for profile in galaxy.cls_list_from(cls=ag.mp.MassProfile)
    ]
    if len(profiles) != 1 or type(profiles[0]) is not ag.mp.Isothermal:
        raise AnalyticHessianRefused(
            f"tracer mass profiles are {[type(p).__name__ for p in profiles]}, need exactly "
            "one Isothermal"
        )
    lens_galaxy = next(g for g in tracer.galaxies if g.cls_list_from(cls=ag.mp.MassProfile))
    if not any(galaxy is lens_galaxy for galaxy in planes[0]):
        raise AnalyticHessianRefused("the Isothermal is not in the image plane")
    return profiles[0]


def _raw(values):
    return values.array if hasattr(values, "array") else values


def _sie_kappa_gamma(sie, grid):
    """kappa and (gamma_2, gamma_1) of an ``Isothermal``, traceable under ``jax.jit``.

    The formulas are ``Isothermal.convergence_2d_from`` / ``shear_yx_2d_from``
    line for line, built from the profile's own geometry helpers with ``xp=jnp``
    threaded through. The library methods themselves cannot be traced with a
    traced ``ell_comps``: ``PowerLawCore.convergence_2d_from`` calls
    ``self.convergence_func(grid_radius=grid_eta)`` without ``xp``, so
    ``einstein_radius_rescaled`` -> ``axis_ratio`` runs ``np.logical_and`` on a
    tracer (``TracerArrayConversionError``), and ``shear_yx_2d_from`` inherits
    it. The Hessian gate checks this function against the library methods
    evaluated eagerly with NumPy.
    """
    transformed = sie.transformed_to_reference_frame_grid_from(grid, jnp)
    ty = _raw(transformed)[:, 0]
    tx = _raw(transformed)[:, 1]
    eta = _raw(sie.elliptical_radii_grid_from(grid=transformed, xp=jnp))
    kappa = sie.einstein_radius_rescaled(jnp) / eta
    r2 = tx**2 + ty**2
    gamma_2 = -2 * kappa * (tx * ty) / r2
    gamma_1 = -kappa * (tx**2 - ty**2) / r2
    shear = sie.rotated_grid_from_reference_frame_from(
        grid=jnp.vstack((gamma_2, gamma_1)).T, xp=jnp, angle=sie.angle(jnp) * 2
    )
    return kappa, _raw(shear)


def _hessian_analytic(self, grid, xp):
    """Closed-form SIE Hessian; (yy, xy, yx, xx) like ``_hessian_via_jax``."""
    _TRACE_COUNTS["analytic"] += 1
    sie = _single_isothermal(self)
    kappa, shear = _sie_kappa_gamma(sie, grid)
    gamma_2 = shear[:, 0]
    gamma_1 = shear[:, 1]
    hessian_xx = kappa + gamma_1
    hessian_yy = kappa - gamma_1
    return (xp.array(hessian_yy), xp.array(gamma_2), xp.array(gamma_2), xp.array(hessian_xx))


def _hessian_jacfwd_counted(self, grid, xp):
    _TRACE_COUNTS["jacfwd"] += 1
    return _ORIGINAL_HESSIAN(self, grid, xp)


_HESSIANS = {
    "jacfwd": _hessian_jacfwd_counted,
    "jacrev": _hessian_jacrev,
    "analytic": _hessian_analytic,
}


@contextlib.contextmanager
def hessian_patch(builder: str):
    """Swap ``LensCalc._hessian_via_jax`` for *builder* inside the scope only.

    ``jacfwd`` installs a counting wrapper around the unpatched original (same
    trace, so the control is production). The original is restored on exit —
    including on an exception — and the restore is asserted.
    """
    if ag.LensCalc.__dict__["_hessian_via_jax"] is not _ORIGINAL_HESSIAN:
        raise AssertionError("LensCalc._hessian_via_jax was not restored by a previous scope")
    before = _TRACE_COUNTS[builder]
    ag.LensCalc._hessian_via_jax = _HESSIANS[builder]
    counter = {"builder": builder, "trace_calls": 0}
    try:
        yield counter
    finally:
        ag.LensCalc._hessian_via_jax = _ORIGINAL_HESSIAN
        counter["trace_calls"] = _TRACE_COUNTS[builder] - before
        if ag.LensCalc.__dict__["_hessian_via_jax"] is not _ORIGINAL_HESSIAN:
            raise AssertionError("LensCalc._hessian_via_jax restore failed")


# ---------------------------------------------------------------------------
# Parameter draws: the timing stream and the PRNGKey gate draws
# ---------------------------------------------------------------------------


def _vector_stream(model, n: int, seed: int) -> np.ndarray:
    """Fixed-seed physical vectors: entry 0 the prior medians, the rest U(0.25,0.75) draws."""
    rng = np.random.default_rng(seed)
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for _ in range(n - 1):
        unit = rng.uniform(0.25, 0.75, size=model.prior_count)
        vectors.append(np.asarray(model.vector_from_unit_vector(unit), dtype=float))
    return np.stack(vectors)


def _gate_vectors(model) -> tuple[list[str], np.ndarray]:
    """Prior medians + one U(0.25,0.75) draw per ``jax.random.PRNGKey(k)``, k = 0..N-1."""
    labels = ["prior_medians"]
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for k in range(N_GATE_KEYS):
        unit = np.asarray(
            jax.random.uniform(
                jax.random.PRNGKey(k), (model.prior_count,), minval=0.25, maxval=0.75
            ),
            dtype=float,
        )
        labels.append(f"PRNGKey({k})")
        vectors.append(np.asarray(model.vector_from_unit_vector(list(unit)), dtype=float))
    return labels, np.stack(vectors)


def _pytrees(model, vectors: np.ndarray) -> list:
    trees = [
        jax.tree_util.tree_map(jnp.asarray, model.instance_from_vector(vector=list(v)))
        for v in vectors
    ]
    treedef = jax.tree_util.tree_structure(trees[0])
    for tree in trees[1:]:
        if jax.tree_util.tree_structure(tree) != treedef:
            raise RuntimeError("the ModelInstance treedef changes across draws")
    return trees


# ---------------------------------------------------------------------------
# Gate 1: Hessian components (analytic and jacrev vs jacfwd)
# ---------------------------------------------------------------------------


def _lens_calc(instance):
    tracer = al.Tracer(galaxies=list(instance.galaxies))
    return ag.LensCalc.from_tracer(tracer=tracer, use_multi_plane=False), tracer


def _hessian_array(builder: str, lens_calc, positions: np.ndarray) -> np.ndarray:
    grid = aa.Grid2DIrregular(values=positions)
    h = _HESSIANS[builder](lens_calc, grid, jnp)
    return np.stack([np.asarray(c, dtype=float) for c in h])  # (4, N): yy, xy, yx, xx


def _tangential_eigenvalue(h: np.ndarray) -> np.ndarray:
    yy, xy, _, xx = h
    return 1.0 - 0.5 * (xx + yy) - np.sqrt((0.5 * (xx - yy)) ** 2 + xy**2)


def _near_critical_positions(lens_calc, centre, positions: np.ndarray) -> tuple:
    """Each position moved along its ray from *centre* onto / around the tangential critical curve."""
    centre = np.asarray(centre, dtype=float)
    offsets = positions - centre
    lo = np.full(len(positions), 0.2)
    hi = np.full(len(positions), 5.0)

    def lam(scale):
        return _tangential_eigenvalue(
            _hessian_array("jacfwd", lens_calc, centre + offsets * scale[:, None])
        )

    lam_lo, lam_hi = lam(lo), lam(hi)
    if not np.all(np.sign(lam_lo) != np.sign(lam_hi)):
        raise RuntimeError("tangential eigenvalue does not change sign along every ray")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lam_mid = lam(mid)
        same = np.sign(lam_mid) == np.sign(lam_lo)
        lo = np.where(same, mid, lo)
        lam_lo = np.where(same, lam_mid, lam_lo)
        hi = np.where(same, hi, mid)
    critical = 0.5 * (lo + hi)
    scales = [critical]
    for step in NEAR_CRITICAL_STEPS:
        scales.append(critical * (1.0 - step))
        scales.append(critical * (1.0 + step))
    stepped = np.concatenate([centre + offsets * s[:, None] for s in scales])
    return stepped, critical, float(np.max(np.abs(lam(critical))))


def _hessian_gate() -> dict:
    model = MODELS["solved"]
    labels, vectors = _gate_vectors(model)
    positions = np.asarray(dataset.positions.array, dtype=float)
    per_builder = {
        b: {"max_rel_err": 0.0, "worst": None, "error": None} for b in ("analytic", "jacrev")
    }
    sets_summary = []
    library_parity = {"max_rel_err": 0.0}
    for label, vector in zip(labels, vectors):
        instance = model.instance_from_vector(vector=list(vector))
        lens_calc, _ = _lens_calc(instance)
        centre = instance.galaxies.lens.mass.centre
        near, critical_scale, lam_residual = _near_critical_positions(lens_calc, centre, positions)
        point_sets = {"dataset": positions, "near_critical": near}
        sie = instance.galaxies.lens.mass
        for pts in point_sets.values():
            grid = aa.Grid2DIrregular(values=pts)
            kappa, shear = _sie_kappa_gamma(sie, grid)
            lib_kappa = np.asarray(_raw(sie.convergence_2d_from(grid=grid, xp=np)))
            lib_shear = np.asarray(_raw(sie.shear_yx_2d_from(grid=grid, xp=np)))
            scale = np.maximum(np.abs(lib_kappa), 1e-300)
            err = max(
                float(np.max(np.abs(np.asarray(kappa) - lib_kappa) / scale)),
                float(np.max(np.abs(np.asarray(shear) - lib_shear) / scale[:, None])),
            )
            library_parity["max_rel_err"] = max(library_parity["max_rel_err"], err)
        for set_name, pts in point_sets.items():
            reference = _hessian_array("jacfwd", lens_calc, pts)
            scale = np.maximum(np.max(np.abs(reference), axis=0), 1e-300)
            for builder in ("analytic", "jacrev"):
                record = per_builder[builder]
                if record["error"] is not None:
                    continue
                try:
                    candidate = _hessian_array(builder, lens_calc, pts)
                except Exception as exc:  # noqa: BLE001 — a refusal is a gate failure
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    continue
                rel = np.abs(candidate - reference) / scale[None, :]
                worst = float(rel.max())
                if worst >= record["max_rel_err"]:
                    idx = np.unravel_index(int(rel.argmax()), rel.shape)
                    record["max_rel_err"] = worst
                    record["worst"] = {
                        "instance": label,
                        "set": set_name,
                        "component": ("yy", "xy", "yx", "xx")[idx[0]],
                        "position": pts[idx[1]].tolist(),
                    }
        sets_summary.append(
            {
                "instance": label,
                "critical_scale": critical_scale.tolist(),
                "tangential_eigenvalue_residual_at_critical": lam_residual,
                "n_dataset": len(positions),
                "n_near_critical": len(near),
            }
        )
    for record in per_builder.values():
        record["pass"] = bool(record["error"] is None and record["max_rel_err"] <= HESSIAN_RTOL)
    library_parity["pass"] = bool(library_parity["max_rel_err"] <= HESSIAN_RTOL)
    library_parity["what"] = (
        "in-cell traceable kappa/gamma vs Isothermal.convergence_2d_from / shear_yx_2d_from "
        "(eager, xp=np), error scaled by |kappa|"
    )
    per_builder["analytic"]["library_formula_parity"] = library_parity
    per_builder["analytic"]["pass"] = bool(
        per_builder["analytic"]["pass"] and library_parity["pass"]
    )
    return {
        "rtol": HESSIAN_RTOL,
        "metric": "max |H_builder - H_jacfwd| / max_components |H_jacfwd| per point",
        "near_critical_steps": list(NEAR_CRITICAL_STEPS),
        "mapping": "H_xx = kappa + gamma_1, H_yy = kappa - gamma_1, H_xy = H_yx = gamma_2 "
        "with shear_yx_2d_from[:, 0] = gamma_2, [:, 1] = gamma_1 (no sign flips)",
        "builders": per_builder,
        "instances": sets_summary,
    }


# ---------------------------------------------------------------------------
# Route factories: FRESH closures (and a fresh AnalysisPoint) per call
# ---------------------------------------------------------------------------


def _grad_factory(lane: str, mode: str):
    def factory():
        analysis = _analysis(FIT_CLS[lane], use_jax=True)

        def log_likelihood(params):
            return analysis.log_likelihood_function(instance=params)

        if mode == "rev":
            return jax.value_and_grad(log_likelihood)

        def with_aux(params):
            value = log_likelihood(params)
            return value, value

        jac = jax.jacfwd(with_aux, has_aux=True)

        def value_and_grad_fwd(params):
            grad, value = jac(params)
            return value, grad

        return value_and_grad_fwd

    return factory


def _forward_factory(lane: str):
    def factory():
        analysis = _analysis(FIT_CLS[lane], use_jax=True)

        def log_likelihood(params):
            return analysis.log_likelihood_function(instance=params)

        return log_likelihood

    return factory


def _split(kind: str, out) -> tuple[np.ndarray, np.ndarray | None]:
    if kind == "grad":
        value, grad = out
        leaves = [np.asarray(leaf, dtype=float).ravel() for leaf in jax.tree_util.tree_leaves(grad)]
        return np.asarray(value, dtype=float), np.concatenate(leaves)
    return np.asarray(out, dtype=float), None


# ---------------------------------------------------------------------------
# Compile a route inside its patch scope
# ---------------------------------------------------------------------------


def _compile_route(route: str, builder: str, factory, example, *, label: str) -> dict:
    """Trace, lower and compile a FRESH closure inside the route's patch scope."""
    jax.clear_caches()
    with hessian_patch(builder) as counter:
        fn = factory()
        t0 = time.perf_counter()
        lowered = jax.jit(fn).lower(example)
        lower_s = time.perf_counter() - t0
        hlo_sha = hashlib.sha256(lowered.as_text().encode()).hexdigest()
        t0 = time.perf_counter()
        compiled = lowered.compile()
        compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    first = block(compiled(example))
    first_call_s = time.perf_counter() - t0
    if counter["trace_calls"] < 1:
        raise AssertionError(f"[{label}/{route}] the {builder} Hessian was never traced")
    if not _THREADS_AFTER_FIRST_COMPILE:
        _THREADS_AFTER_FIRST_COMPILE["threads"] = _thread_count()
        _THREADS_AFTER_FIRST_COMPILE["label"] = f"{label}/{route}"
    print(
        f"  [{label}/{route}] lower {lower_s * 1e3:8.1f} ms  compile {compile_s * 1e3:8.1f} ms  "
        f"first {first_call_s * 1e3:8.3f} ms  (hessian={builder}, traced x{counter['trace_calls']})"
    )
    return {
        "executable": compiled,
        "fn": fn,
        "first": first,
        "record": {
            "hessian": builder,
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "first_call_s": float(first_call_s),
            "flops": _flops(compiled),
            "hessian_trace_calls": int(counter["trace_calls"]),
            "stablehlo_sha256": hlo_sha,
        },
    }


# ---------------------------------------------------------------------------
# Gates 2 + 3 for one row: route agreement, finite/non-zero, eager == JIT
# ---------------------------------------------------------------------------


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _grad_close(g: np.ndarray, ref: np.ndarray) -> tuple[bool, float]:
    atol = GRAD_ATOL_SCALE * float(np.max(np.abs(ref)))
    ok = bool(np.all(np.abs(g - ref) <= atol + GRAD_RTOL * np.abs(ref)))
    return ok, _rel(g, ref)


def _route_gate(kind, routes, compiled, gate_trees, gate_labels, eager_idx, label) -> dict:
    control = CONTROL[kind]
    outputs = {}
    for route in routes:
        if route not in compiled:
            continue
        exe = compiled[route]["executable"]
        outputs[route] = [_split(kind, block(exe(tree))) for tree in gate_trees]
    ref = outputs.get(control)
    gate = {}
    for route in routes:
        if route not in compiled:
            gate[route] = {"pass": False, "reason": compiled_errors[label][route]}
            continue
        reasons = []
        worst_value = 0.0
        worst_grad = 0.0
        min_grad_l2 = None
        for i, (value, grad) in enumerate(outputs[route]):
            if not np.all(np.isfinite(value)):
                reasons.append(f"{gate_labels[i]}: non-finite log L")
            if ref is not None:
                rv = _rel(value, ref[i][0])
                worst_value = max(worst_value, rv)
                if rv > VALUE_RTOL:
                    reasons.append(f"{gate_labels[i]}: log L rel err {rv:.2e} > {VALUE_RTOL}")
            if grad is not None:
                l2 = float(np.linalg.norm(grad))
                min_grad_l2 = l2 if min_grad_l2 is None else min(min_grad_l2, l2)
                if not np.all(np.isfinite(grad)):
                    reasons.append(f"{gate_labels[i]}: non-finite gradient")
                if not l2 > 0.0:
                    reasons.append(f"{gate_labels[i]}: gradient identically zero")
                if ref is not None:
                    ok, rg = _grad_close(grad, ref[i][1])
                    worst_grad = max(worst_grad, rg)
                    if not ok:
                        reasons.append(f"{gate_labels[i]}: gradient rel err {rg:.2e} > {GRAD_RTOL}")
        # eager == JIT, inside the route's patch scope, no jit anywhere
        builder = compiled[route]["record"]["hessian"]
        worst_eager_value = 0.0
        worst_eager_grad = 0.0
        try:
            with hessian_patch(builder), jax.disable_jit():
                for i in eager_idx:
                    value, grad = _split(kind, compiled[route]["fn"](gate_trees[i]))
                    jv, jg = outputs[route][i]
                    ev = _rel(value, jv)
                    worst_eager_value = max(worst_eager_value, ev)
                    if ev > VALUE_RTOL:
                        reasons.append(f"eager {gate_labels[i]}: log L rel err {ev:.2e}")
                    if grad is not None:
                        ok, eg = _grad_close(grad, jg)
                        worst_eager_grad = max(worst_eager_grad, eg)
                        if not ok:
                            reasons.append(f"eager {gate_labels[i]}: gradient rel err {eg:.2e}")
        except Exception as exc:  # noqa: BLE001 — reported as a gate failure
            reasons.append(f"eager evaluation raised {type(exc).__name__}: {exc}")
        gate[route] = {
            "pass": not reasons,
            "reasons": reasons[:20],
            "n_instances": len(gate_trees),
            "max_rel_err_log_l_vs_control": worst_value,
            "max_rel_err_grad_vs_control": worst_grad if kind == "grad" else None,
            "min_grad_l2": min_grad_l2,
            "eager_instances": [gate_labels[i] for i in eager_idx],
            "max_rel_err_eager_vs_jit_log_l": worst_eager_value,
            "max_rel_err_eager_vs_jit_grad": worst_eager_grad if kind == "grad" else None,
        }
    # No route may have been served another's trace: StableHLO hashes pairwise distinct.
    hashes = {r: compiled[r]["record"]["stablehlo_sha256"] for r in routes if r in compiled}
    if len(set(hashes.values())) != len(hashes):
        raise AssertionError(f"[{label}] two routes lowered to identical StableHLO: {hashes}")
    return gate


# ---------------------------------------------------------------------------
# One row: compile all routes, gate, time the passing ones interleaved
# ---------------------------------------------------------------------------

compiled_errors: dict[str, dict] = {}


def _run_row(lane: str, kind: str, stream_trees, gate_trees, gate_labels, hessian_ok) -> dict:
    label = f"{lane}_{kind}"
    print(f"\n--- row {label} ---")
    spec = GRAD_ROUTES if kind == "grad" else {r: ("forward", b) for r, b in FORWARD_ROUTES.items()}
    routes = tuple(spec)
    compiled = {}
    compiled_errors[label] = {}
    for route, (mode, builder) in spec.items():
        factory = _grad_factory(lane, mode) if kind == "grad" else _forward_factory(lane)
        try:
            compiled[route] = _compile_route(route, builder, factory, stream_trees[0], label=label)
        except Exception as exc:  # noqa: BLE001 — a route that cannot trace fails the gate
            compiled_errors[label][route] = f"compile raised {type(exc).__name__}: {exc}"
            print(f"  [{label}/{route}] FAILED to compile: {compiled_errors[label][route]}")
    if CONTROL[kind] not in compiled:
        raise RuntimeError(f"[{label}] the control route failed: {compiled_errors[label]}")

    eager_idx = [0, 1, 2][: len(gate_trees)]
    gate = _route_gate(kind, routes, compiled, gate_trees, gate_labels, eager_idx, label)
    for route, (_, builder) in spec.items():
        if builder != "jacfwd" and not hessian_ok[builder]:
            gate[route]["pass"] = False
            gate[route].setdefault("reasons", []).append(
                f"Hessian gate for {builder} failed (see hessian_gate)"
            )
    timed = tuple(r for r in routes if gate[r]["pass"])
    for route in routes:
        status = "PASS" if gate[route]["pass"] else "FAIL -> not timed"
        detail = (
            ""
            if gate[route]["pass"]
            else f" {gate[route].get('reason') or gate[route].get('reasons', [])[:3]}"
        )
        print(f"  gate {route:<13} {status}{detail}")
    if CONTROL[kind] not in timed:
        raise RuntimeError(f"[{label}] the control route failed its own gate: {gate}")

    executables = {route: compiled[route]["executable"] for route in timed}
    n_args = len(stream_trees)
    warm_s = {route: [] for route in timed}
    for route in timed:
        for w in range(N_WARM):
            t0 = time.perf_counter()
            block(executables[route](stream_trees[w % n_args]))
            warm_s[route].append(time.perf_counter() - t0)

    times = {route: [] for route in timed}
    per_round = {route: [] for route in timed}
    call_index = 0
    for r in range(N_ROUNDS):
        order = timed[r % len(timed) :] + timed[: r % len(timed)]
        start = call_index
        for route in order:
            round_times = []
            for c in range(N_CALLS):
                k = (start + c) % n_args
                t0 = time.perf_counter()
                block(executables[route](stream_trees[k]))
                round_times.append(time.perf_counter() - t0)
            times[route].extend(round_times)
            per_round[route].append(float(np.median(round_times)))
        call_index = start + N_CALLS

    control = CONTROL[kind]
    stats = {route: _stats_ms(times[route]) for route in timed}
    vs_control = {}
    for i, route in enumerate(timed):
        if route == control:
            continue
        paired = np.asarray(per_round[route]) / np.asarray(per_round[control])
        vs_control[route] = {
            "ratio_route_over_control": _median_ratio(
                times[route], times[control], BOOTSTRAP_SEED + i
            ),
            "saved_ms": _median_saving_ms(times[control], times[route], BOOTSTRAP_SEED + 100 + i),
            "paired_round_ratio": {
                "median": float(np.median(paired)),
                "p10": float(np.percentile(paired, 10)),
                "p90": float(np.percentile(paired, 90)),
            },
        }
    cs = stats[control]
    row = {
        "label": label,
        "lane": lane,
        "kind": kind,
        "control": control,
        "routes": {r: {"mode": spec[r][0], "hessian": spec[r][1]} for r in routes},
        "timed_routes": list(timed),
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "route_order": "round-robin, start rotated each round",
        "gate": gate,
        "compile": {route: compiled[route]["record"] for route in compiled},
        "compile_errors": compiled_errors[label],
        "per_call_ms": {route: [t * 1e3 for t in times[route]] for route in timed},
        "warm_ms": {route: [t * 1e3 for t in warm_s[route]] for route in timed},
        "stats": stats,
        "vs_control": vs_control,
        "minimum_detectable_improvement": float((cs["p90_ms"] - cs["p10_ms"]) / cs["median_ms"]),
    }
    print(
        "  median ms  " + "  ".join(f"{route} {stats[route]['median_ms']:.4f}" for route in timed)
    )
    for route, v in vs_control.items():
        rr, sv = v["ratio_route_over_control"], v["saved_ms"]
        print(
            f"    {route:<13} /{control} {rr['ratio']:.3f} [{rr['ci90_low']:.3f}, "
            f"{rr['ci90_high']:.3f}]  saved {sv['saved_ms']:+.4f} ms "
            f"[{sv['ci90_low']:+.4f}, {sv['ci90_high']:+.4f}]"
        )
    return row


# ===========================================================================
# Run
# ===========================================================================

print("=== gate 1: Hessian components (analytic / jacrev vs jacfwd) ===")
hessian_gate = _hessian_gate()
for builder, record in hessian_gate["builders"].items():
    print(
        f"  {builder:<9} {'PASS' if record['pass'] else 'FAIL'}  max rel err "
        f"{record['max_rel_err']:.3e}  worst {record['worst']}  error {record['error']}"
    )
HESSIAN_OK = {b: r["pass"] for b, r in hessian_gate["builders"].items()}

rows: dict[str, dict] = {}
free_parameters: dict[str, int] = {}
for lane_index, lane in enumerate(LANES):
    model = MODELS[lane]
    free_parameters[lane] = int(model.prior_count)
    stream_trees = _pytrees(model, _vector_stream(model, N_STREAM, SEED + lane_index))
    gate_labels, gate_vectors = _gate_vectors(model)
    gate_trees = _pytrees(model, gate_vectors)
    for kind in ("grad", "forward"):
        rows[f"{lane}_{kind}"] = _run_row(
            lane, kind, stream_trees, gate_trees, gate_labels, HESSIAN_OK
        )


def _phase2c_rule() -> dict:
    per_lane = {}
    for lane in LANES:
        row = rows[f"{lane}_grad"]
        control_ms = row["stats"]["rev"]["median_ms"]
        per_route = {}
        for route in GRAD_ROUTES:
            if route == "rev":
                continue
            gate_green = bool(row["gate"][route]["pass"])
            if route not in row["vs_control"]:
                per_route[route] = {
                    "gate_green": gate_green,
                    "go": False,
                    "timed": False,
                    "verdict": NO_GO,
                    "verdict_reason": "not timed: its correctness gate is red",
                }
                continue
            v = row["vs_control"][route]
            saved, ratio = v["saved_ms"], v["ratio_route_over_control"]
            fraction = saved["saved_ms"] / control_ms
            point = bool(saved["saved_ms"] >= GO_MIN_SAVED_MS and fraction >= GO_MIN_FRACTION)
            ci = bool(
                saved["ci90_low"] >= GO_MIN_SAVED_MS and ratio["ci90_high"] <= 1.0 - GO_MIN_FRACTION
            )
            verdict = ab_rule_verdict(
                [
                    Criterion(
                        "saved_ms",
                        saved["saved_ms"],
                        saved["ci90_low"],
                        saved["ci90_high"],
                        GO_MIN_SAVED_MS,
                        AT_LEAST,
                        "ms",
                    ),
                    Criterion(
                        "ratio_route_over_rev",
                        ratio["ratio"],
                        ratio["ci90_low"],
                        ratio["ci90_high"],
                        1.0 - GO_MIN_FRACTION,
                        AT_MOST,
                    ),
                ],
                n=row["n_rounds"],
                min_n=MIN_AB_ROUNDS,
                confidence=AB_CONFIDENCE,
                gates={"correctness": gate_green},
            )
            per_route[route] = {
                "timed": True,
                "rev_ms": control_ms,
                "route_ms": row["stats"][route]["median_ms"],
                "saved_ms": saved["saved_ms"],
                "saved_ms_ci90": [saved["ci90_low"], saved["ci90_high"]],
                "fraction_saved": fraction,
                "ratio": ratio["ratio"],
                "ratio_ci90": [ratio["ci90_low"], ratio["ci90_high"]],
                "point_estimate_clears_bar": point,
                "ci_excludes_bar": ci,
                "gate_green": gate_green,
                "verdict": verdict.verdict,
                "verdict_reason": verdict.reason,
                "resolvable_effect": {c.name: c.mdi for c in verdict.criteria},
                "go": verdict.go,
            }
        per_lane[lane] = per_route
    return {
        "rule": f"GO iff saved >= {GO_MIN_SAVED_MS} ms AND >= {GO_MIN_FRACTION:.0%} vs rev on "
        "the value_and_grad-equivalent call, 90% CIs excluding the bar (saved CI low >= "
        f"{GO_MIN_SAVED_MS} ms, ratio CI high <= {1 - GO_MIN_FRACTION:.2f}), correctness gate green; "
        "NO_GO only when an interval is wholly on the wrong side or the gate is red; INCONCLUSIVE "
        f"otherwise or below {MIN_AB_ROUNDS} rounds (shared ab_rule_verdict, #362)",
        "effective_n": "n_rounds (the interval is still an iid call bootstrap; #362 fix phase 4)",
        "decides_on": "hpc_ral_cpu_fp64 only",
        "per_lane": per_lane,
    }


phase2c_rule = _phase2c_rule()
config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_device_fp64"
)
_home = str(Path.home())


def _scrub(obj):
    """Replace the home directory in every string (no /home/<user> in a committed JSON)."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str) and _home and _home != "/":
        return obj.replace(_home, "~")
    return obj


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


device = device_info_dict()
if device.get("jax_compilation_cache_dir"):
    device["jax_compilation_cache_dir"] = (
        "<scrubbed>/" + Path(device["jax_compilation_cache_dir"]).name
    )

summary = {
    "cell": "backward_pass_ab",
    "issue": "PyAutoLabs/autolens_profiling#325",
    "config_name": config_name,
    "quick": QUICK,
    "instrument": INSTRUMENT,
    "library_versions": {
        "autolens": al.__version__,
        "autoarray": aa.__version__,
        "autofit": af.__version__,
    },
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    "source_revisions": source_revisions(_ROOT),
    "device": device,
    "jax_devices": [str(d) for d in jax.devices()],
    "machine": machine_info_dict(),
    "cpu_model": _cpu_model(),
    "thread_environment": thread_environment(),
    "sched_affinity": len(os.sched_getaffinity(0)),
    "os_cpu_count": os.cpu_count(),
    "xla_flags": os.environ.get("XLA_FLAGS"),
    "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "loadavg_start": list(_LOADAVG_START),
    "loadavg_end": list(os.getloadavg()),
    "xla_threads_observed_after_first_compile": _THREADS_AFTER_FIRST_COMPILE,
    "threads_at_end": _thread_count(),
    "protocol": {
        "grad_routes": {r: {"mode": m, "hessian": b} for r, (m, b) in GRAD_ROUTES.items()},
        "forward_routes": FORWARD_ROUTES,
        "lanes": {
            "solved": "PointSolved + FitPositionsSourceSolved",
            "plain": "PointFlux + FitPositionsSource",
        },
        "argument": "registered ModelInstance pytree (tree_map(jnp.asarray, instance_from_vector)) "
        "for every route; fwd = jax.jacfwd over its scalar leaves (the flat parameter vector)",
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "n_stream": N_STREAM,
        "instance_seed": SEED,
        "instance_stream": "entry 0 = prior medians; rest = vector_from_unit_vector(U(0.25,0.75))",
        "gate_draws": f"prior medians + PRNGKey 0..{N_GATE_KEYS - 1} U(0.25,0.75) unit draws",
        "gate_tolerances": {
            "hessian_rtol": HESSIAN_RTOL,
            "log_l_rtol": VALUE_RTOL,
            "grad_rtol": GRAD_RTOL,
            "grad_atol": f"{GRAD_ATOL_SCALE} x max|grad_control|",
        },
        "hessian_patch": "scoped monkeypatch of LensCalc._hessian_via_jax; trace/lower/compile "
        "inside the scope; restore asserted; trace-call counter > 0; StableHLO hashes distinct",
        "timed_object": "jax.jit(fn).lower(example).compile() executable",
        "cache_trap_guard": "fresh closure + fresh AnalysisPoint per route; jax.clear_caches()",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": "90%"},
        "solver": "solver=None; FitPositionsSource never invokes a PointSolver",
        "pytrees": "autofit.jax.register_model(model) for both lanes before any trace",
    },
    "free_parameters": free_parameters,
    "hessian_gate": hessian_gate,
    "phase2c_rule": phase2c_rule,
    "rows": rows,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_source",
    default_basename="backward_pass_ab_local",
    cell="backward_pass_ab",
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2))

# ---------------------------------------------------------------------------
# PNG: per-row boxplots
# ---------------------------------------------------------------------------
palette = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974"]
fig, axes = plt.subplots(
    nrows=len(LANES), ncols=2, figsize=(11.0, 4.8 * len(LANES)), constrained_layout=True
)
for i, lane in enumerate(LANES):
    for j, kind in enumerate(("grad", "forward")):
        ax = axes[i, j]
        row = rows[f"{lane}_{kind}"]
        timed = row["timed_routes"]
        box = ax.boxplot(
            [row["per_call_ms"][r] for r in timed],
            whis=(10, 90),
            showfliers=False,
            patch_artist=True,
        )
        for patch, colour in zip(box["boxes"], palette):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(timed) + 1))
        ax.set_xticklabels(timed, fontsize=8, rotation=15)
        ax.set_ylabel("ms per call")
        lines = [f"{lane} {kind} (control {row['control']})"]
        for route, v in row["vs_control"].items():
            rr = v["ratio_route_over_control"]
            lines.append(
                f"{route}: {rr['ratio']:.2f}x [{rr['ci90_low']:.2f}, {rr['ci90_high']:.2f}], "
                f"saved {v['saved_ms']['saved_ms']:+.3f} ms"
            )
        ax.set_title("\n".join(lines), fontsize=7)
        ax.set_ylim(bottom=0)
fig.suptitle(
    f"Source-plane point-source backward-pass A/B ({config_name}); whiskers p10/p90, "
    f"rounds={N_ROUNDS}, calls/round={N_CALLS}",
    fontsize=10,
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 100)
print(f"BACKWARD-PASS A/B  ({config_name}, rounds={N_ROUNDS}, calls/round={N_CALLS})")
print("=" * 100)
print(f"  {'row':<16} {'route':<13} {'median':>9} {'p10':>9} {'p90':>9} {'compile s':>10}")
for label, row in rows.items():
    for route in row["timed_routes"]:
        s = row["stats"][route]
        print(
            f"  {label:<16} {route:<13} {s['median_ms']:9.4f} {s['p10_ms']:9.4f} "
            f"{s['p90_ms']:9.4f} {row['compile'][route]['compile_s']:10.3f}"
        )
print("-" * 100)
for lane, per_route in phase2c_rule["per_lane"].items():
    for route, verdict in per_route.items():
        if not verdict.get("timed"):
            print(f"  phase-2c rule [{lane}/{route}]: NOT TIMED (gate failed) -> NO_GO")
            continue
        print(
            f"  phase-2c rule [{lane}/{route}]: saved {verdict['saved_ms']:+.4f} ms "
            f"[{verdict['saved_ms_ci90'][0]:+.4f}, {verdict['saved_ms_ci90'][1]:+.4f}] = "
            f"{verdict['fraction_saved']:.1%}; ratio {verdict['ratio']:.3f} "
            f"[{verdict['ratio_ci90'][0]:.3f}, {verdict['ratio_ci90'][1]:.3f}] -> "
            f"{verdict['verdict']} ({config_name}; RAL CPU decides)"
        )
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
