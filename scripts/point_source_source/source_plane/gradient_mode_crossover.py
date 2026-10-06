"""
JAX Profiling: Source-Plane Point-Source Gradient-Mode Crossover (phase 2c)
==========================================================================

Where does forward-mode stop beating reverse-mode for the gradient of the
production source-plane point-source likelihood? (autolens_profiling #329,
single-source only.) Phase 2b (``backward_pass_ab.py``) found that
``jax.jacfwd`` over the flat parameter vector cuts the ``value_and_grad``
call by 24-46 % at 5 free parameters. Forward mode costs one JVP per free
parameter, reverse mode a roughly constant multiple of the forward call, so
forward mode must lose above some crossover ``n*``. This cell measures the ratio
``fwd / rev`` along a model-complexity ladder on the same seeded ``simple``
dataset and interpolates ``n*``. No library is edited.

Model ladder (solved-lane free parameters; the plain lane adds the free source
centre and the ``PointFlux`` flux, +3)
-----------------------------------------------------------------------------

- ``L5``  — ``Isothermal`` (phase-2b baseline, identical priors).
- ``L7``  — + ``ExternalShear``.
- ``L9``  — + ``PowerLawMultipole`` m=4, free ``multipole_comps``; its centre and
  Einstein radius are linked to the main lens (af prior linking) and its slope is
  fixed at 2.0 (the SIE value; ``PowerLawMultipole``'s default).
- ``L11`` — + m=3 multipole comps (same linking).
- ``L16`` — + a satellite ``Isothermal`` at the lens redshift, centre prior at
  (y, x) = (-1.0, 1.0) (1.4" from the lens, >= 1.49" from every image), small
  Einstein radius (0.1").
- ``L19`` — main lens ``Isothermal`` -> ``PowerLaw`` with free slope (the
  multipoles' slopes are linked to it), + an ``ExternalShear`` on the satellite
  galaxy. (An m=1 multipole was the alternative: ``PowerLawMultipole`` m=1 is
  singular at slope exactly 2 — deflections ``-inf`` / NaN at the SIE prior
  median — so it cannot sit on an SIE-centred slope prior. The satellite shear
  is exactly degenerate with the main shear in deflection; that is harmless for
  timing and for the gradient gate.)
- ``L24`` — extension beyond the planned ladder: + a second satellite
  ``Isothermal`` at the lens redshift, centre prior (y, x) = (1.0, -1.0) (1.4"
  from the lens, >= 1.54" from every image), theta_E 0.1". Added because fwd
  still wins at ``L19`` on the laptop lead run, so the crossing is bracketed
  rather than extrapolated past the last rung.

Every extra component is centred on (almost) zero perturbation. The prior means
are 1e-3, NOT exactly 0: ``ExternalShear`` and ``PowerLawMultipole`` are
parameterised by a magnitude ``sqrt(c_0^2 + c_1^2)`` and an angle
``arctan2(c_1, c_0)``, whose gradient at exactly (0, 0) is NaN (and the
satellite ``Isothermal``'s ``ell_comps`` likewise); at 1e-3 the gradient is
finite.

Argument handling (production)
------------------------------

Every route takes the flat physical parameter vector (a ``jnp`` array of
``model.prior_count`` entries) and builds the instance INSIDE the trace with
``model.instance_from_vector(vector, xp=jnp)`` — what PyAutoFit's JAX
``Fitness.call`` does. This matters from ``L9`` on: linked priors make the
``ModelInstance`` pytree carry more leaves than free parameters (e.g. 12 leaves
for 9 parameters), so ``jacfwd`` over the pytree (phase 2b's argument, where
leaves == parameters) would push one tangent per leaf, not per parameter.

Routes and call shapes (per rung x lane)
----------------------------------------

- ``rev`` — ``jax.value_and_grad(ll)``.
- ``fwd`` — ``jax.jacfwd(ll, has_aux=True)`` with the value as aux; one tangent
  per free parameter.

Each route is timed as a ``single`` call and as a ``batched`` call
``jax.jit(jax.vmap(route))`` over B = 8 parameter vectors (PyAutoFit's
multi-start gradient search vmaps ``value_and_grad`` over its starts,
``multi_start_gradient/search.py:1089``).

Correctness gate (before timing; a failing rung is reported, never timed)
-------------------------------------------------------------------------

Over the prior medians + ``PRNGKey`` 0..15 draws (``U(0.25, 0.75)`` unit cube ->
``vector_from_unit_vector``): ``fwd`` log L equals ``rev`` to rtol 1e-10 and its
gradient to rtol 1e-8 (atol 1e-12 x max|grad|); both finite and non-zero (L2);
the batched routes equal the single ``rev`` row by row (same tolerances); eager
(``jax.disable_jit()``) == JIT for the single routes on the prior medians and
``PRNGKey`` 0 and 1; the four executables' StableHLO hashes pairwise distinct.
``autofit.jax.register_model(model)`` is applied to every model (memory grad0).

Timing
------

Per executable: lower / compile / first-call seconds and XLA flops. Then >= 3
warm calls and ``--rounds`` rounds x ``--calls`` calls, the four executables
(rev/fwd x single/batched) interleaved with the order rotated each round, the
parameters walking a fixed-seed 16-vector (single) / 16-batch (batched) stream,
every call ``block_until_ready``'d. Ratio ``fwd / rev`` per call shape with a
bootstrap 90 % CI (2000 resamples of the per-call times).

Crossover
---------

Per lane x call shape, ``n*`` is the ``n_params`` where the ratio first rises
through 1, linear in ``n`` between the bracketing rungs. Its interval comes from
the per-rung bootstrap ratio samples (curve j uses resample j of every rung):
the 5-95 % range of the crossings, with the fraction of curves that cross. With
no crossing inside the ladder the record says so (``fwd wins through n=<max>``
or ``rev wins from n=<min>``) and gives a least-squares linear extrapolation of
the ratio over the last three rungs.

Output
------

``results/breakdown/point_source_source/gradient_mode_crossover_<config_name>.{json,png}``.
No top-level ``autolens_version`` (versions under ``library_versions``): a
note-backed measurement, not a README auto-table row.
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

from likelihood_breakdown.provenance import source_revisions, thread_environment  # noqa: E402
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

RUNG_NAMES = ("L5", "L7", "L9", "L11", "L16", "L19", "L24")

_cli = parse_profile_cli()
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--rounds", type=int, default=None)
_cell_parser.add_argument("--calls", type=int, default=None)
_cell_parser.add_argument("--n-stream", type=int, default=16)
_cell_parser.add_argument("--warm", type=int, default=3)
_cell_parser.add_argument("--seed", type=int, default=329)
_cell_parser.add_argument("--batch", type=int, default=8, help="vmap batch size B")
_cell_parser.add_argument("--gate-keys", type=int, default=16, help="PRNGKey 0..N-1 draws")
_cell_parser.add_argument(
    "--rungs", default=",".join(RUNG_NAMES), help="comma-separated subset of the ladder"
)
_cell_parser.add_argument("--quick", action="store_true", help="rounds=5, calls=5")
_args = _cli.parse_cell_args(_cell_parser)

QUICK = bool(_args.quick)
N_ROUNDS = _args.rounds if _args.rounds is not None else (5 if QUICK else 20)
N_CALLS = _args.calls if _args.calls is not None else (5 if QUICK else 20)
N_STREAM = _args.n_stream
N_WARM = max(3, _args.warm)
SEED = _args.seed
BATCH = _args.batch
N_GATE_KEYS = _args.gate_keys
RUNGS = tuple(r.strip() for r in _args.rungs.split(",") if r.strip())
if N_ROUNDS < 1 or N_CALLS < 1 or N_STREAM < 1 or N_GATE_KEYS < 1 or BATCH < 1:
    raise SystemExit("--rounds, --calls, --n-stream, --gate-keys and --batch must be positive")
if not RUNGS or any(r not in RUNG_NAMES for r in RUNGS):
    raise SystemExit(f"--rungs must be a subset of {RUNG_NAMES}")

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402

INSTRUMENT = "simple"
LANES = ("solved", "plain")
MODES = ("rev", "fwd")
SHAPES = ("single", "batched")
ROUTES = tuple(f"{m}_{s}" for s in SHAPES for m in MODES)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
VALUE_RTOL = 1.0e-10
GRAD_RTOL = 1.0e-8
GRAD_ATOL_SCALE = 1.0e-12
#: Prior mean of every added perturbation component (see the module docstring).
PERTURBATION_MEAN = 1.0e-3
PERTURBATION_SIGMA = 0.01
SATELLITE_CENTRE = (-1.0, 1.0)
SATELLITE_2_CENTRE = (1.0, -1.0)
SATELLITE_EINSTEIN_RADIUS = 0.1

_THREADS_AFTER_FIRST_COMPILE: dict = {}


def _thread_count() -> int | None:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


# ---------------------------------------------------------------------------
# The model ladder
# ---------------------------------------------------------------------------


def _gauss(mean: float, sigma: float):
    return af.GaussianPrior(mean=mean, sigma=sigma)


def _main_mass(power_law: bool):
    """The phase-2b SIE priors; ``PowerLaw`` adds a free slope centred on 2 (SIE)."""
    mass = af.Model(al.mp.PowerLaw if power_law else al.mp.Isothermal)
    mass.centre.centre_0 = _gauss(0.0, 0.005)
    mass.centre.centre_1 = _gauss(0.0, 0.005)
    mass.einstein_radius = _gauss(1.6, 0.05)
    mass.ell_comps.ell_comps_0 = _gauss(0.05263158, 0.01)
    mass.ell_comps.ell_comps_1 = _gauss(0.0, 0.01)
    if power_law:
        mass.slope = _gauss(2.0, 0.05)
    return mass


def _multipole(mass, order: int, power_law: bool):
    """``PowerLawMultipole`` of order *order*, linked to the main lens; comps free."""
    multipole = af.Model(al.mp.PowerLawMultipole, m=order)
    multipole.centre = mass.centre
    multipole.einstein_radius = mass.einstein_radius
    multipole.slope = mass.slope if power_law else 2.0
    multipole.multipole_comps.multipole_comps_0 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
    multipole.multipole_comps.multipole_comps_1 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
    return multipole


def _ladder_model(rung: str, *, solved: bool):
    n = int(rung[1:])
    power_law = n >= 19
    mass = _main_mass(power_law)
    components = {"mass": mass}
    if n >= 7:
        shear = af.Model(al.mp.ExternalShear)
        shear.gamma_1 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
        shear.gamma_2 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
        components["shear"] = shear
    if n >= 9:
        components["multipole_4"] = _multipole(mass, 4, power_law)
    if n >= 11:
        components["multipole_3"] = _multipole(mass, 3, power_law)
    galaxies = {"lens": af.Model(al.Galaxy, redshift=0.5, **components)}
    if n >= 16:
        satellite = af.Model(al.mp.Isothermal)
        satellite.centre.centre_0 = _gauss(SATELLITE_CENTRE[0], 0.01)
        satellite.centre.centre_1 = _gauss(SATELLITE_CENTRE[1], 0.01)
        satellite.einstein_radius = _gauss(SATELLITE_EINSTEIN_RADIUS, 0.01)
        satellite.ell_comps.ell_comps_0 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
        satellite.ell_comps.ell_comps_1 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
        satellite_components = {"mass": satellite}
        if n >= 19:
            satellite_shear = af.Model(al.mp.ExternalShear)
            satellite_shear.gamma_1 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
            satellite_shear.gamma_2 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
            satellite_components["shear"] = satellite_shear
        galaxies["satellite"] = af.Model(al.Galaxy, redshift=0.5, **satellite_components)
    if n >= 24:
        satellite_2 = af.Model(al.mp.Isothermal)
        satellite_2.centre.centre_0 = _gauss(SATELLITE_2_CENTRE[0], 0.01)
        satellite_2.centre.centre_1 = _gauss(SATELLITE_2_CENTRE[1], 0.01)
        satellite_2.einstein_radius = _gauss(SATELLITE_EINSTEIN_RADIUS, 0.01)
        satellite_2.ell_comps.ell_comps_0 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
        satellite_2.ell_comps.ell_comps_1 = _gauss(PERTURBATION_MEAN, PERTURBATION_SIGMA)
        galaxies["satellite_2"] = af.Model(al.Galaxy, redshift=0.5, mass=satellite_2)
    if solved:
        point = af.Model(al.ps.PointSolved)
    else:
        point = af.Model(al.ps.PointFlux)
        point.centre.centre_0 = _gauss(0.07, 0.005)
        point.centre.centre_1 = _gauss(0.07, 0.005)
    galaxies["source"] = af.Model(al.Galaxy, redshift=1.0, point_0=point)
    return af.Collection(galaxies=af.Collection(**galaxies))


RUNG_COMPONENTS = {
    "L5": "Isothermal",
    "L7": "Isothermal + ExternalShear",
    "L9": "+ PowerLawMultipole m=4 (comps free; centre/theta_E linked, slope 2)",
    "L11": "+ PowerLawMultipole m=3 (comps free, linked)",
    "L16": "+ satellite Isothermal at z=0.5 (centre (-1, 1), theta_E 0.1)",
    "L19": "main Isothermal -> PowerLaw (free slope, multipole slopes linked) + satellite shear",
    "L24": "+ second satellite Isothermal at z=0.5 (centre (1, -1), theta_E 0.1) [extension]",
}

dataset_path = _ROOT / "dataset" / "point_source" / INSTRUMENT
auto_simulate_if_missing(
    dataset_path,
    dataset_type="point_source",
    instrument=INSTRUMENT,
    workspace_root=_ROOT,
)
dataset = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")
FIT_CLS = {"solved": al.FitPositionsSourceSolved, "plain": al.FitPositionsSource}


def _analysis(lane: str):
    return al.AnalysisPoint(
        dataset=dataset, solver=None, fit_positions_cls=FIT_CLS[lane], use_jax=True
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _stats_ms(seconds, scale: float = 1.0) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1.0e3 / scale
    return {
        "n": int(ms.size),
        "median_ms": float(np.median(ms)),
        "p10_ms": float(np.percentile(ms, 10)),
        "p90_ms": float(np.percentile(ms, 90)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "mean_ms": float(ms.mean()),
    }


def _ratio_boots(numerator, denominator, seed: int) -> tuple[dict, np.ndarray]:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        boots[i] = np.median(rng.choice(num, num.size)) / np.median(rng.choice(den, den.size))
    return {
        "ratio": float(np.median(num) / np.median(den)),
        "ci90_low": float(np.percentile(boots, 5)),
        "ci90_high": float(np.percentile(boots, 95)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }, boots


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
# Parameter draws
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


def _batches(vectors: np.ndarray, n_batches: int, seed: int) -> np.ndarray:
    """``n_batches`` batches of B vectors, each a fixed-seed permutation-cycle of *vectors*."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        idx = rng.choice(len(vectors), size=BATCH, replace=len(vectors) < BATCH)
        out.append(vectors[idx])
    return np.stack(out)


# ---------------------------------------------------------------------------
# Route factories: FRESH closures (and a fresh AnalysisPoint) per executable
# ---------------------------------------------------------------------------


def _route_factory(model, lane: str, mode: str, shape: str, trace_counter: dict):
    def factory():
        analysis = _analysis(lane)

        def log_likelihood(vector):
            trace_counter["n"] += 1
            instance = model.instance_from_vector(vector=vector, xp=jnp)
            return analysis.log_likelihood_function(instance=instance)

        if mode == "rev":
            fn = jax.value_and_grad(log_likelihood)
        else:

            def with_aux(vector):
                value = log_likelihood(vector)
                return value, value

            jac = jax.jacfwd(with_aux, has_aux=True)

            def fn(vector):
                grad, value = jac(vector)
                return value, grad

        return jax.vmap(fn) if shape == "batched" else fn

    return factory


def _compile_route(route: str, factory, example, counter: dict, *, label: str) -> dict:
    jax.clear_caches()
    before = counter["n"]
    fn = factory()
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(example)
    lower_s = time.perf_counter() - t0
    hlo_sha = hashlib.sha256(lowered.as_text().encode()).hexdigest()
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    block(compiled(example))
    first_call_s = time.perf_counter() - t0
    traces = counter["n"] - before
    if traces < 1:
        raise AssertionError(f"[{label}/{route}] the likelihood was never traced")
    if not _THREADS_AFTER_FIRST_COMPILE:
        _THREADS_AFTER_FIRST_COMPILE["threads"] = _thread_count()
        _THREADS_AFTER_FIRST_COMPILE["label"] = f"{label}/{route}"
    print(
        f"  [{label}/{route:<11}] lower {lower_s:6.2f} s  compile {compile_s:6.2f} s  "
        f"first {first_call_s * 1e3:9.3f} ms"
    )
    return {
        "executable": compiled,
        "fn": fn,
        "record": {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "first_call_s": float(first_call_s),
            "flops": _flops(compiled),
            "likelihood_traces": int(traces),
            "stablehlo_sha256": hlo_sha,
        },
    }


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _grad_close(g, ref) -> tuple[bool, float]:
    g = np.asarray(g, dtype=float)
    ref = np.asarray(ref, dtype=float)
    atol = GRAD_ATOL_SCALE * float(np.max(np.abs(ref)))
    ok = bool(np.all(np.abs(g - ref) <= atol + GRAD_RTOL * np.abs(ref)))
    return ok, _rel(g, ref)


def _np_out(out) -> tuple[np.ndarray, np.ndarray]:
    value, grad = out
    return np.asarray(value, dtype=float), np.asarray(grad, dtype=float)


def _gate(compiled: dict, gate_vectors: np.ndarray, gate_labels: list[str], label: str) -> dict:
    """fwd vs rev (single), batched vs single rev, finite/non-zero, eager == JIT, HLO hashes."""
    n = len(gate_vectors)
    single = {}
    for mode in MODES:
        exe = compiled[f"{mode}_single"]["executable"]
        single[mode] = [_np_out(block(exe(jnp.asarray(v)))) for v in gate_vectors]
    ref = single["rev"]
    # batched: pad the gate vectors to a multiple of B by cycling, unpad the rows
    n_pad = -(-n // BATCH) * BATCH
    order = np.arange(n_pad) % n
    batched = {}
    for mode in MODES:
        exe = compiled[f"{mode}_batched"]["executable"]
        values, grads = [], []
        for start in range(0, n_pad, BATCH):
            v, g = _np_out(block(exe(jnp.asarray(gate_vectors[order[start : start + BATCH]]))))
            values.append(v)
            grads.append(g)
        values = np.concatenate(values)[:n]
        grads = np.concatenate(grads)[:n]
        batched[mode] = [(values[i], grads[i]) for i in range(n)]

    outputs = {
        "rev_single": single["rev"],
        "fwd_single": single["fwd"],
        "rev_batched": batched["rev"],
        "fwd_batched": batched["fwd"],
    }
    gate = {}
    for route, outs in outputs.items():
        reasons = []
        worst_value = 0.0
        worst_grad = 0.0
        min_l2 = None
        for i, (value, grad) in enumerate(outs):
            if not np.all(np.isfinite(value)):
                reasons.append(f"{gate_labels[i]}: non-finite log L")
            if not np.all(np.isfinite(grad)):
                reasons.append(f"{gate_labels[i]}: non-finite gradient")
            l2 = float(np.linalg.norm(grad))
            min_l2 = l2 if min_l2 is None else min(min_l2, l2)
            if not l2 > 0.0:
                reasons.append(f"{gate_labels[i]}: gradient identically zero")
            rv = _rel(value, ref[i][0])
            worst_value = max(worst_value, rv)
            if rv > VALUE_RTOL:
                reasons.append(f"{gate_labels[i]}: log L rel err {rv:.2e} > {VALUE_RTOL}")
            ok, rg = _grad_close(grad, ref[i][1])
            worst_grad = max(worst_grad, rg)
            if not ok:
                reasons.append(f"{gate_labels[i]}: gradient rel err {rg:.2e} > {GRAD_RTOL}")
        gate[route] = {
            "pass": not reasons,
            "reasons": reasons[:20],
            "n_instances": n,
            "max_rel_err_log_l_vs_rev_single": worst_value,
            "max_rel_err_grad_vs_rev_single": worst_grad,
            "min_grad_l2": min_l2,
        }
    # eager == JIT (single routes; the batched routes are pinned to rev_single above)
    eager_idx = [0, 1, 2][:n]
    for mode in MODES:
        route = f"{mode}_single"
        worst_ev, worst_eg = 0.0, 0.0
        reasons = gate[route]["reasons"]
        try:
            with jax.disable_jit():
                for i in eager_idx:
                    value, grad = _np_out(compiled[route]["fn"](jnp.asarray(gate_vectors[i])))
                    jv, jg = outputs[route][i]
                    ev = _rel(value, jv)
                    worst_ev = max(worst_ev, ev)
                    if ev > VALUE_RTOL:
                        reasons.append(f"eager {gate_labels[i]}: log L rel err {ev:.2e}")
                    ok, eg = _grad_close(grad, jg)
                    worst_eg = max(worst_eg, eg)
                    if not ok:
                        reasons.append(f"eager {gate_labels[i]}: gradient rel err {eg:.2e}")
        except Exception as exc:  # noqa: BLE001 — reported as a gate failure
            reasons.append(f"eager evaluation raised {type(exc).__name__}: {exc}")
        gate[route]["eager_instances"] = [gate_labels[i] for i in eager_idx]
        gate[route]["max_rel_err_eager_vs_jit_log_l"] = worst_ev
        gate[route]["max_rel_err_eager_vs_jit_grad"] = worst_eg
        gate[route]["pass"] = not reasons
    hashes = {r: compiled[r]["record"]["stablehlo_sha256"] for r in ROUTES}
    distinct = len(set(hashes.values())) == len(hashes)
    return {
        "routes": gate,
        "stablehlo_hashes_distinct": distinct,
        "pass": bool(distinct and all(g["pass"] for g in gate.values())),
    }


# ---------------------------------------------------------------------------
# One rung x lane: compile, gate, time interleaved
# ---------------------------------------------------------------------------


def _run_cell(rung: str, lane: str, lane_index: int, rung_index: int) -> tuple[dict, dict]:
    label = f"{rung}_{lane}"
    model = _ladder_model(rung, solved=lane == "solved")
    register_model_pytrees(model)
    n_params = int(model.prior_count)
    medians_instance = model.instance_from_vector(
        vector=list(model.physical_values_from_prior_medians)
    )
    n_leaves = len(jax.tree_util.tree_leaves(medians_instance))
    print(f"\n--- {label}: n_params {n_params} (instance pytree leaves {n_leaves}) ---")
    seed = SEED + 100 * rung_index + lane_index
    stream = _vector_stream(model, N_STREAM, seed)
    batch_stream = _batches(stream, N_STREAM, seed + 7)
    gate_labels, gate_vectors = _gate_vectors(model)

    counter = {"n": 0}
    compiled = {}
    compile_errors = {}
    for shape in SHAPES:
        example = jnp.asarray(stream[0] if shape == "single" else batch_stream[0])
        for mode in MODES:
            route = f"{mode}_{shape}"
            factory = _route_factory(model, lane, mode, shape, counter)
            try:
                compiled[route] = _compile_route(route, factory, example, counter, label=label)
            except Exception as exc:  # noqa: BLE001 — a failing rung is reported, not timed
                compile_errors[route] = f"compile raised {type(exc).__name__}: {exc}"
                print(f"  [{label}/{route}] FAILED to compile: {compile_errors[route]}")
    base = {
        "rung": rung,
        "lane": lane,
        "components": RUNG_COMPONENTS[rung],
        "n_params": n_params,
        "instance_pytree_leaves": n_leaves,
        "compile": {r: c["record"] for r, c in compiled.items()},
        "compile_errors": compile_errors,
    }
    if compile_errors:
        base["gate"] = {"pass": False, "reason": "compile failed"}
        base["timed"] = False
        return base, {}
    gate = _gate(compiled, gate_vectors, gate_labels, label)
    base["gate"] = gate
    for route, g in gate["routes"].items():
        status = "PASS" if g["pass"] else f"FAIL {g['reasons'][:3]}"
        print(
            f"  gate {route:<11} {status}  logL {g['max_rel_err_log_l_vs_rev_single']:.1e}  "
            f"grad {g['max_rel_err_grad_vs_rev_single']:.1e}  min|g| {g['min_grad_l2']:.3g}"
        )
    if not gate["pass"]:
        print(f"  [{label}] GATE FAILED -> not timed")
        base["timed"] = False
        return base, {}

    executables = {r: compiled[r]["executable"] for r in ROUTES}
    args = {
        "single": [jnp.asarray(v) for v in stream],
        "batched": [jnp.asarray(b) for b in batch_stream],
    }
    shape_of = {r: r.split("_", 1)[1] for r in ROUTES}
    warm_s = {r: [] for r in ROUTES}
    for route in ROUTES:
        for w in range(N_WARM):
            t0 = time.perf_counter()
            block(executables[route](args[shape_of[route]][w % N_STREAM]))
            warm_s[route].append(time.perf_counter() - t0)

    times = {r: [] for r in ROUTES}
    call_index = 0
    for r in range(N_ROUNDS):
        order = ROUTES[r % len(ROUTES) :] + ROUTES[: r % len(ROUTES)]
        start = call_index
        for route in order:
            stream_args = args[shape_of[route]]
            exe = executables[route]
            for c in range(N_CALLS):
                arg = stream_args[(start + c) % N_STREAM]
                t0 = time.perf_counter()
                block(exe(arg))
                times[route].append(time.perf_counter() - t0)
        call_index = start + N_CALLS

    stats = {r: _stats_ms(times[r]) for r in ROUTES}
    per_vector = {r: _stats_ms(times[r], BATCH) for r in ROUTES if shape_of[r] == "batched"}
    ratios = {}
    boots = {}
    for j, shape in enumerate(SHAPES):
        ratio, sample = _ratio_boots(
            times[f"fwd_{shape}"],
            times[f"rev_{shape}"],
            BOOTSTRAP_SEED + 1000 * rung_index + 10 * lane_index + j,
        )
        ratios[shape] = ratio
        boots[shape] = sample
    base.update(
        {
            "timed": True,
            "n_rounds": N_ROUNDS,
            "n_calls_per_round": N_CALLS,
            "n_warm": N_WARM,
            "batch": BATCH,
            "per_call_ms": {r: [round(t * 1e3, 6) for t in times[r]] for r in ROUTES},
            "warm_ms": {r: [t * 1e3 for t in warm_s[r]] for r in ROUTES},
            "stats_per_call": stats,
            "stats_per_vector_batched": per_vector,
            "ratio_fwd_over_rev": ratios,
            "minimum_detectable_improvement_rev_single": float(
                (stats["rev_single"]["p90_ms"] - stats["rev_single"]["p10_ms"])
                / stats["rev_single"]["median_ms"]
            ),
        }
    )
    print(
        "  median ms/call  "
        + "  ".join(f"{r} {stats[r]['median_ms']:.4f}" for r in ROUTES)
        + f"  | ratio fwd/rev single {ratios['single']['ratio']:.3f} "
        f"[{ratios['single']['ci90_low']:.3f}, {ratios['single']['ci90_high']:.3f}]"
        f"  batched {ratios['batched']['ratio']:.3f} "
        f"[{ratios['batched']['ci90_low']:.3f}, {ratios['batched']['ci90_high']:.3f}]"
    )
    return base, boots


# ---------------------------------------------------------------------------
# Crossover estimate
# ---------------------------------------------------------------------------


def _first_crossing(ns: np.ndarray, rs: np.ndarray):
    """(kind, n*): 'cross' with the interpolated n, 'rev_from_start', or 'none'."""
    if rs[0] >= 1.0:
        return "rev_from_start", None
    for i in range(len(ns) - 1):
        if rs[i] < 1.0 <= rs[i + 1]:
            return "cross", float(ns[i] + (1.0 - rs[i]) * (ns[i + 1] - ns[i]) / (rs[i + 1] - rs[i]))
    return "none", None


def _extrapolate(ns: np.ndarray, rs: np.ndarray) -> dict:
    k = min(3, len(ns))
    if k < 2:
        return {"fit_rungs_n": ns.tolist(), "slope_per_param": None, "n_at_ratio_1": None}
    slope, intercept = np.polyfit(ns[-k:], rs[-k:], 1)
    n_one = float((1.0 - intercept) / slope) if slope > 0 else None
    return {
        "fit_rungs_n": ns[-k:].tolist(),
        "slope_per_param": float(slope),
        "intercept": float(intercept),
        "n_at_ratio_1": n_one,
        "what": "least-squares line through the last three rungs' point ratios",
    }


def _crossover(cells: dict, all_boots: dict) -> dict:
    out = {}
    for lane in LANES:
        timed = [c for c in cells.values() if c["lane"] == lane and c.get("timed")]
        timed.sort(key=lambda c: c["n_params"])
        if len(timed) < 1:
            out[lane] = {"status": "no timed rungs"}
            continue
        ns = np.array([c["n_params"] for c in timed], dtype=float)
        out[lane] = {}
        for shape in SHAPES:
            rs = np.array([c["ratio_fwd_over_rev"][shape]["ratio"] for c in timed])
            boot_matrix = np.stack([all_boots[f"{c['rung']}_{lane}"][shape] for c in timed])
            kind, n_star = _first_crossing(ns, rs)
            crossings = []
            kinds = {"cross": 0, "rev_from_start": 0, "none": 0}
            for j in range(boot_matrix.shape[1]):
                kj, nj = _first_crossing(ns, boot_matrix[:, j])
                kinds[kj] += 1
                if nj is not None:
                    crossings.append(nj)
            record = {
                "n_params": ns.tolist(),
                "ratio": rs.tolist(),
                "point": kind,
                "n_star": n_star,
                "bootstrap_curves": int(boot_matrix.shape[1]),
                "bootstrap_fraction": {k: v / boot_matrix.shape[1] for k, v in kinds.items()},
                "n_star_ci90": (
                    [float(np.percentile(crossings, 5)), float(np.percentile(crossings, 95))]
                    if crossings
                    else None
                ),
                "extrapolation": _extrapolate(ns, rs),
            }
            if kind == "cross":
                record["summary"] = f"n* = {n_star:.1f}"
            elif kind == "rev_from_start":
                record["summary"] = f"rev wins from n={int(ns[0])}"
            else:
                record["summary"] = f"fwd wins through n={int(ns[-1])}"
            out[lane][shape] = record
    return out


# ===========================================================================
# Run
# ===========================================================================

cells: dict[str, dict] = {}
all_boots: dict[str, dict] = {}
for rung_index, rung in enumerate(RUNGS):
    for lane_index, lane in enumerate(LANES):
        cell, boots = _run_cell(rung, lane, lane_index, RUNG_NAMES.index(rung))
        cells[f"{rung}_{lane}"] = cell
        if boots:
            all_boots[f"{rung}_{lane}"] = boots

crossover = _crossover(cells, all_boots)
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

ladder = {
    rung: {
        "components": RUNG_COMPONENTS[rung],
        "n_params": {lane: cells[f"{rung}_{lane}"]["n_params"] for lane in LANES},
        "instance_pytree_leaves": {
            lane: cells[f"{rung}_{lane}"]["instance_pytree_leaves"] for lane in LANES
        },
        "gate_pass": {lane: bool(cells[f"{rung}_{lane}"]["gate"]["pass"]) for lane in LANES},
    }
    for rung in RUNGS
}

summary = {
    "cell": "gradient_mode_crossover",
    "issue": "PyAutoLabs/autolens_profiling#329",
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
        "routes": {
            "rev": "jax.value_and_grad(ll)",
            "fwd": "jax.jacfwd(ll, has_aux=True), value as aux, one tangent per free parameter",
        },
        "shapes": {
            "single": "jax.jit(route)",
            "batched": f"jax.jit(jax.vmap(route)) over B={BATCH} vectors "
            "(multi_start_gradient/search.py:1089)",
        },
        "argument": "flat physical vector (jnp, model.prior_count entries); "
        "model.instance_from_vector(vector, xp=jnp) inside the trace (autofit JAX Fitness.call)",
        "lanes": {
            "solved": "PointSolved + FitPositionsSourceSolved",
            "plain": "PointFlux + FitPositionsSource (free source centre + flux, +3 params)",
        },
        "perturbation_prior": f"Gaussian mean {PERTURBATION_MEAN}, sigma {PERTURBATION_SIGMA} "
        "(not 0: the magnitude/angle parameterisation has a NaN gradient at exactly (0, 0))",
        "satellite": {
            "centre_prior_mean": list(SATELLITE_CENTRE),
            "satellite_2_centre_prior_mean": list(SATELLITE_2_CENTRE),
            "einstein_radius_prior_mean": SATELLITE_EINSTEIN_RADIUS,
        },
        "rungs": list(RUNGS),
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "n_stream": N_STREAM,
        "instance_seed": SEED,
        "instance_stream": "entry 0 = prior medians; rest = vector_from_unit_vector(U(0.25,0.75)); "
        "batched stream = 16 fixed-seed batches of B draws from it",
        "route_order": "4 executables round-robin, start rotated each round",
        "gate_draws": f"prior medians + PRNGKey 0..{N_GATE_KEYS - 1} U(0.25,0.75) unit draws",
        "gate_tolerances": {
            "log_l_rtol": VALUE_RTOL,
            "grad_rtol": GRAD_RTOL,
            "grad_atol": f"{GRAD_ATOL_SCALE} x max|grad_rev_single|",
        },
        "timed_object": "jax.jit(fn).lower(example).compile() executable",
        "cache_trap_guard": "fresh closure + fresh AnalysisPoint per executable; "
        "jax.clear_caches(); StableHLO hashes pairwise distinct per rung x lane",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": "90%"},
        "crossover": "first upward crossing of ratio = 1, linear in n between bracketing rungs; "
        "interval = 5-95% of bootstrap-curve crossings (curve j = resample j of every rung)",
        "solver": "solver=None; FitPositionsSource never invokes a PointSolver",
        "pytrees": "autofit.jax.register_model(model) for every model before any trace",
    },
    "ladder": ladder,
    "crossover": crossover,
    "cells": cells,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_source",
    default_basename="gradient_mode_crossover_local",
    cell="gradient_mode_crossover",
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2))

# ---------------------------------------------------------------------------
# PNG: ratio fwd/rev vs n_params, CI bands, single vs batched, both lanes
# ---------------------------------------------------------------------------
colours = {"solved": "#4C72B0", "plain": "#C44E52"}
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True, sharey=True)
for ax, shape in zip(axes, SHAPES):
    for lane in LANES:
        timed = sorted(
            (c for c in cells.values() if c["lane"] == lane and c.get("timed")),
            key=lambda c: c["n_params"],
        )
        if not timed:
            continue
        ns = [c["n_params"] for c in timed]
        r = [c["ratio_fwd_over_rev"][shape]["ratio"] for c in timed]
        lo = [c["ratio_fwd_over_rev"][shape]["ci90_low"] for c in timed]
        hi = [c["ratio_fwd_over_rev"][shape]["ci90_high"] for c in timed]
        cx = crossover[lane][shape]
        ax.plot(ns, r, "o-", color=colours[lane], label=f"{lane}: {cx['summary']}")
        ax.fill_between(ns, lo, hi, color=colours[lane], alpha=0.25)
        if cx["point"] == "cross":
            ax.axvline(cx["n_star"], color=colours[lane], ls=":", lw=1)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xlabel("free parameters n_params")
    ax.set_title(f"{shape} call" + (f" (vmap B={BATCH})" if shape == "batched" else ""))
    ax.legend(fontsize=8)
axes[0].set_ylabel("median time ratio fwd / rev (90% CI)")
fig.suptitle(
    f"Source-plane point-source gradient-mode crossover ({config_name}); "
    f"rounds={N_ROUNDS}, calls/round={N_CALLS}",
    fontsize=10,
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 100)
print(f"GRADIENT-MODE CROSSOVER  ({config_name}, rounds={N_ROUNDS}, calls/round={N_CALLS})")
print("=" * 100)
print(
    f"  {'cell':<12} {'n':>3} {'gate':>5} {'rev_1':>9} {'fwd_1':>9} {'ratio_1':>8} "
    f"{'rev_B':>9} {'fwd_B':>9} {'ratio_B':>8} {'cmp rev/fwd s':>14}"
)
for label, c in cells.items():
    if not c.get("timed"):
        print(f"  {label:<12} {c['n_params']:>3}  FAIL (not timed)")
        continue
    s, rr = c["stats_per_call"], c["ratio_fwd_over_rev"]
    print(
        f"  {label:<12} {c['n_params']:>3} {'ok':>5} {s['rev_single']['median_ms']:9.4f} "
        f"{s['fwd_single']['median_ms']:9.4f} {rr['single']['ratio']:8.3f} "
        f"{s['rev_batched']['median_ms']:9.4f} {s['fwd_batched']['median_ms']:9.4f} "
        f"{rr['batched']['ratio']:8.3f} "
        f"{c['compile']['rev_single']['compile_s']:6.2f}/{c['compile']['fwd_single']['compile_s']:<6.2f}"
    )
print("-" * 100)
for lane, per_shape in crossover.items():
    for shape, cx in per_shape.items():
        if not isinstance(cx, dict):
            continue
        print(
            f"  crossover [{lane}/{shape}]: {cx['summary']}  ci90 {cx['n_star_ci90']}  "
            f"boot {cx['bootstrap_fraction']}  extrap n@1 {cx['extrapolation'].get('n_at_ratio_1')}"
        )
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
