"""
JAX Profiling: Source-Plane Point-Source gradient_mode Through the Library (phase 2e)
=====================================================================================

Does the phase-2b/2c forward-mode speed-up of the source-plane point-source
gradient arrive through the REAL library path? (autolens_profiling #334,
single-source only.) Phase 2d merged an analysis-declared ``gradient_mode``
(PyAutoFit#1649, PyAutoLens#752): ``al.AnalysisPoint.gradient_mode = "forward"``
and ``MultiStartGradient(gradient_mode=...)`` overrides it. This cell drives
``af.MultiStartAdam`` itself on the phase-2c ``L5`` and ``L24`` rungs (same
seeded ``simple`` dataset, same ladder model builders, both lanes) and times the
batched objective step the search builds, in forward vs reverse mode. No library
is edited.

How the real path is exercised
------------------------------

``MultiStartGradient._fit`` builds its objective inline (it is not exposed):
``Fitness(fom_is_log_likelihood=False, convert_to_chi_squared=True, ...)`` ->
``value_and_grad_from(fitness.call, gradient_mode)`` -> the local
``_value_and_grad_finite`` wrapper (adds ``all(isfinite(grad))`` and the model
constraint as outputs) -> ``_vmapped = jax.jit(jax.vmap(_value_and_grad_finite))``.
This cell does NOT rebuild that chain. It runs ``search.fit(model, analysis)``
with ``jax.jit`` temporarily wrapped by a recorder: the moment the search jits a
function named ``_value_and_grad_finite`` the recorder keeps the returned jitted
object and aborts the fit (a private exception, before any compile or likelihood
evaluation). The kept object IS the search's ``_vmapped`` for that mode, closing
over the search's own ``Fitness``. The objective is therefore ``-2 log posterior``
(``Fitness.call``), not phase 2c's bare ``log L``; the log-prior term is a few
flops.

- ``forward`` runs use ``af.MultiStartAdam()`` with NO override, so the mode
  comes from ``AnalysisPoint``'s declaration; ``reverse`` runs pass
  ``gradient_mode="reverse"``. The search's own log line
  (``MultiStartGradient gradient mode: <mode> (<source>).``) is captured and
  checked, and ``search._resolved_gradient_mode(analysis)`` is recorded.

Correctness gate (before timing; a failing rung x lane is reported, never timed)
------------------------------------------------------------------------------

- Declared default: ``af.MultiStartAdam()._resolved_gradient_mode(AnalysisPoint)``
  is ``"forward"``, the ``gradient_mode="reverse"`` override resolves to
  ``"reverse"``, and the captured log lines say the same with the right source.
- Per rung x lane: over one start batch of B = 8 per ``jax.random.PRNGKey(k)``,
  k = 0..15 (``U(0.25, 0.75)`` unit cube -> ``vector_from_unit_vector``), the
  forward ``_vmapped`` equals the reverse one: objective rtol 1e-10, gradient
  rtol 1e-8 (atol 1e-12 x max|grad|); every objective and gradient finite, the
  search's own ``grad_finite`` flag True, every gradient non-zero (L2); the two
  executables' StableHLO hashes distinct.
- End to end, per lane at ``L5``: one short ``MultiStartAdam`` fit per mode
  (``n_starts=8``, ``n_steps=20``, fixed ``seed``), the full ``search.fit``
  (no abort), giving the same best vector (rtol 1e-6) and max log likelihood
  (rtol 1e-8). A throwaway warm-up fit runs first so one-time process costs
  (font cache, first output write) land in neither mode. Wall recorded per mode:
  the whole ``search.fit`` and the ``_fit`` call alone (compiles + steps, no
  output I/O). On a GPU ``_fit`` also pays the search's batched-memory probe
  (two throwaway compiles), in both modes alike.

Timing
------

Per mode: ``jax.clear_caches()``, then ``_vmapped.lower(example)`` /
``.compile()`` seconds, first call and XLA flops. Four routes are timed:
``<mode>_exe`` (the AOT-compiled executable, comparable to phase 2c's batched
row) and ``<mode>_jit`` (the captured jitted callable itself, dispatched exactly
as the search's step loop calls it; its own compile happens in the warm-up and
is not timed). >= 3 warm calls, then ``--rounds`` x ``--calls`` calls, the four
routes interleaved with the order rotated each round, the batch argument walking
phase 2c's fixed-seed 16-batch stream (same seeds, so the same start batches),
every call ``block_until_ready``'d. Ratio forward / reverse per route kind with a
bootstrap 90 % CI (2000 resamples). The phase-2c batched ratio for the same host
is read from ``results/breakdown/point_source_source/gradient_mode_crossover_
<config_name>.json`` when present and recorded next to it.

Output
------

``results/breakdown/point_source_source/gradient_mode_library_ab_<config_name>.{json,png}``.
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
import logging
import shutil
import tempfile

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib
import numpy as np

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from likelihood_breakdown.provenance import source_revisions, thread_environment  # noqa: E402
from likelihood_breakdown.round_bootstrap import round_median_ratio  # noqa: E402
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

#: Phase 2c's ladder, kept only so the per-rung seeds (and so the start
#: batches) are byte-identical to ``gradient_mode_crossover.py``'s.
RUNG_NAMES_2C = ("L5", "L7", "L9", "L11", "L16", "L19", "L24")
RUNG_NAMES = ("L5", "L24")

_cli = parse_profile_cli()
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--rounds", type=int, default=None)
_cell_parser.add_argument("--calls", type=int, default=None)
_cell_parser.add_argument("--n-stream", type=int, default=16)
_cell_parser.add_argument("--warm", type=int, default=3)
_cell_parser.add_argument("--seed", type=int, default=329, help="phase 2c's instance seed")
_cell_parser.add_argument("--batch", type=int, default=8, help="n_starts = vmap batch size B")
_cell_parser.add_argument("--gate-keys", type=int, default=16, help="PRNGKey 0..N-1 batches")
_cell_parser.add_argument("--fit-steps", type=int, default=20, help="end-to-end fit n_steps")
_cell_parser.add_argument("--fit-seed", type=int, default=334, help="end-to-end fit seed")
_cell_parser.add_argument(
    "--rungs", default=",".join(RUNG_NAMES), help="comma-separated subset of L5,L24"
)
_cell_parser.add_argument("--no-fit", action="store_true", help="skip the end-to-end fits")
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
FIT_STEPS = _args.fit_steps
FIT_SEED = _args.fit_seed
RUN_FITS = not _args.no_fit
RUNGS = tuple(r.strip() for r in _args.rungs.split(",") if r.strip())
if N_ROUNDS < 1 or N_CALLS < 1 or N_STREAM < 1 or N_GATE_KEYS < 1 or BATCH < 1:
    raise SystemExit("--rounds, --calls, --n-stream, --gate-keys and --batch must be positive")
if FIT_STEPS < 1:
    raise SystemExit("--fit-steps must be positive")
if not RUNGS or any(r not in RUNG_NAMES for r in RUNGS):
    raise SystemExit(f"--rungs must be a subset of {RUNG_NAMES}")

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402
from autofit.jax.gradient import resolve_gradient_mode  # noqa: E402

INSTRUMENT = "simple"
LANES = ("solved", "plain")
MODES = ("forward", "reverse")
KINDS = ("exe", "jit")
ROUTES = tuple(f"{m}_{k}" for k in KINDS for m in MODES)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345
VALUE_RTOL = 1.0e-10
GRAD_RTOL = 1.0e-8
GRAD_ATOL_SCALE = 1.0e-12
FIT_VECTOR_RTOL = 1.0e-6
FIT_VECTOR_ATOL = 1.0e-12
FIT_LOG_L_RTOL = 1.0e-8
#: Prior mean of every added perturbation component (see gradient_mode_crossover.py).
PERTURBATION_MEAN = 1.0e-3
PERTURBATION_SIGMA = 0.01
SATELLITE_CENTRE = (-1.0, 1.0)
SATELLITE_2_CENTRE = (1.0, -1.0)
SATELLITE_EINSTEIN_RADIUS = 0.1
MODE_LOG_PREFIX = "MultiStartGradient gradient mode:"

_THREADS_AFTER_FIRST_COMPILE: dict = {}

# Search output (pre-fit files, the end-to-end fits' zips) goes to a throwaway
# directory, never into the repo's output/.
_SEARCH_OUTPUT = Path(tempfile.mkdtemp(prefix="gradient_mode_library_ab_"))
af.conf.instance.output_path = str(_SEARCH_OUTPUT)


class _ModeLogHandler(logging.Handler):
    """Collects the search's ``MultiStartGradient gradient mode: ...`` log lines."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        message = record.getMessage()
        if MODE_LOG_PREFIX in message:
            self.lines.append(message)


_MODE_LOG = _ModeLogHandler()
_root_logger = logging.getLogger()
_root_logger.addHandler(_MODE_LOG)
if _root_logger.level > logging.INFO or _root_logger.level == logging.NOTSET:
    _root_logger.setLevel(logging.INFO)


def _thread_count() -> int | None:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


# ---------------------------------------------------------------------------
# The model ladder (verbatim from gradient_mode_crossover.py; L5 and L24 used)
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
    "L24": "Isothermal -> PowerLaw + shear + m=4/m=3 multipoles (linked) + 2 satellite "
    "Isothermals (first with shear) -- phase 2c's L24",
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


def _search(name: str, mode: str, n_steps: int, seed: int):
    """``forward`` = the declared default (no override); ``reverse`` = the override."""
    kwargs = {} if mode == "forward" else {"gradient_mode": "reverse"}
    return af.MultiStartAdam(name=name, n_starts=BATCH, n_steps=n_steps, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Statistics (as gradient_mode_crossover.py)
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


def _ratio_boots(numerator, denominator, seed: int) -> dict:
    """median(numerator) / median(denominator), with a paired whole-round bootstrap 90 % interval.

    Resamples whole rounds with the same indices for both arms (#362 fix phase 4,
    the shared ``likelihood_breakdown.round_bootstrap``); ``effective_n`` is
    ``N_ROUNDS``. It replaced an iid, unpaired resampling of individual calls.
    """
    return round_median_ratio(
        numerator, denominator, n_rounds=N_ROUNDS, seed=seed, samples=BOOTSTRAP_SAMPLES
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
# Parameter draws (phase 2c's streams, same seeds)
# ---------------------------------------------------------------------------


def _vector_stream(model, n: int, seed: int) -> np.ndarray:
    """Fixed-seed physical vectors: entry 0 the prior medians, the rest U(0.25,0.75) draws."""
    rng = np.random.default_rng(seed)
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for _ in range(n - 1):
        unit = rng.uniform(0.25, 0.75, size=model.prior_count)
        vectors.append(np.asarray(model.vector_from_unit_vector(unit), dtype=float))
    return np.stack(vectors)


def _batches(vectors: np.ndarray, n_batches: int, seed: int) -> np.ndarray:
    """``n_batches`` batches of B vectors, each a fixed-seed draw from *vectors*."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        idx = rng.choice(len(vectors), size=BATCH, replace=len(vectors) < BATCH)
        out.append(vectors[idx])
    return np.stack(out)


def _gate_batches(model) -> np.ndarray:
    """One B-start batch per ``jax.random.PRNGKey(k)``: U(0.25,0.75) unit cube -> physical."""
    out = []
    for k in range(N_GATE_KEYS):
        unit = np.asarray(
            jax.random.uniform(
                jax.random.PRNGKey(k), (BATCH, model.prior_count), minval=0.25, maxval=0.75
            ),
            dtype=float,
        )
        out.append(np.stack([model.vector_from_unit_vector(list(u)) for u in unit]))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Capture the search's own batched step
# ---------------------------------------------------------------------------


class _Captured(Exception):
    """Raised by the ``jax.jit`` recorder once the search has built ``_vmapped``."""


def _capture_vmapped(model, lane: str, mode: str, label: str) -> dict:
    """Run ``search.fit`` until it jits ``_value_and_grad_finite``; return that jitted object."""
    analysis = _analysis(lane)
    search = _search(f"capture_{label}_{mode}", mode, n_steps=1, seed=SEED)
    resolved = search._resolved_gradient_mode(analysis)
    held: dict = {}
    real_jit = jax.jit

    def recorder(fn, *args, **kwargs):
        out = real_jit(fn, *args, **kwargs)
        if getattr(fn, "__name__", None) == "_value_and_grad_finite":
            held["vmapped"] = out
            held["jit_kwargs"] = sorted(kwargs)
            raise _Captured
        return out

    n_log_before = len(_MODE_LOG.lines)
    jax.jit = recorder
    try:
        search.fit(model=model, analysis=analysis)
    except _Captured:
        pass
    finally:
        jax.jit = real_jit
    if "vmapped" not in held:
        raise AssertionError(
            f"[{label}/{mode}] search.fit never jitted _value_and_grad_finite "
            "(the MultiStartGradient objective build changed?)"
        )
    return {
        "vmapped": held["vmapped"],
        "resolved": resolved,
        "log_lines": _MODE_LOG.lines[n_log_before:],
        "search_gradient_mode_attr": getattr(search, "gradient_mode", None),
        "scaler": type(search.scaler).__name__,
        "bijector": type(search.bijector).__name__,
        "clipper": type(search.clipper).__name__,
        "batch_size": search.batch_size,
    }


def _compile(mode: str, vmapped, example, label: str) -> dict:
    jax.clear_caches()
    t0 = time.perf_counter()
    lowered = vmapped.lower(example)
    lower_s = time.perf_counter() - t0
    hlo_sha = hashlib.sha256(lowered.as_text().encode()).hexdigest()
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    block(compiled(example))
    first_call_s = time.perf_counter() - t0
    if not _THREADS_AFTER_FIRST_COMPILE:
        _THREADS_AFTER_FIRST_COMPILE["threads"] = _thread_count()
        _THREADS_AFTER_FIRST_COMPILE["label"] = f"{label}/{mode}"
    print(
        f"  [{label}/{mode:<7}] lower {lower_s:6.2f} s  compile {compile_s:7.2f} s  "
        f"first {first_call_s * 1e3:9.3f} ms"
    )
    return {
        "executable": compiled,
        "record": {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "lower_plus_compile_s": float(lower_s + compile_s),
            "first_call_s": float(first_call_s),
            "flops": _flops(compiled),
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


def _gate(executables: dict, gate_batches: np.ndarray, hashes: dict) -> dict:
    outs = {}
    for mode in MODES:
        outs[mode] = [
            [np.asarray(x) for x in block(executables[mode](jnp.asarray(b)))] for b in gate_batches
        ]
    reasons = []
    worst_value = 0.0
    worst_grad = 0.0
    min_l2 = None
    for k in range(len(gate_batches)):
        f_fom, f_grad, f_fin, _ = outs["forward"][k]
        r_fom, r_grad, r_fin, _ = outs["reverse"][k]
        for mode, (fom, grad, fin, _v) in (
            ("forward", outs["forward"][k]),
            ("reverse", outs["reverse"][k]),
        ):
            if not np.all(np.isfinite(fom)):
                reasons.append(f"PRNGKey({k}) {mode}: non-finite objective")
            if not np.all(np.isfinite(grad)):
                reasons.append(f"PRNGKey({k}) {mode}: non-finite gradient")
            if not np.all(fin):
                reasons.append(f"PRNGKey({k}) {mode}: search grad_finite flag False")
            l2 = np.linalg.norm(grad, axis=1)
            min_l2 = float(l2.min()) if min_l2 is None else min(min_l2, float(l2.min()))
            if not np.all(l2 > 0.0):
                reasons.append(f"PRNGKey({k}) {mode}: a gradient row is identically zero")
        rv = _rel(f_fom, r_fom)
        worst_value = max(worst_value, rv)
        if rv > VALUE_RTOL:
            reasons.append(f"PRNGKey({k}): objective rel err {rv:.2e} > {VALUE_RTOL}")
        for row in range(f_grad.shape[0]):
            ok, rg = _grad_close(f_grad[row], r_grad[row])
            worst_grad = max(worst_grad, rg)
            if not ok:
                reasons.append(f"PRNGKey({k}) row {row}: gradient rel err {rg:.2e} > {GRAD_RTOL}")
    distinct = hashes["forward"] != hashes["reverse"]
    if not distinct:
        reasons.append("forward and reverse StableHLO hashes are identical")
    return {
        "pass": not reasons,
        "reasons": reasons[:20],
        "n_batches": int(len(gate_batches)),
        "batch": BATCH,
        "max_rel_err_objective_fwd_vs_rev": worst_value,
        "max_rel_err_grad_fwd_vs_rev": worst_grad,
        "min_grad_l2": min_l2,
        "stablehlo_hashes_distinct": bool(distinct),
    }


# ---------------------------------------------------------------------------
# One rung x lane: capture, compile, gate, time interleaved
# ---------------------------------------------------------------------------


def _phase_2c_batched(rung: str, lane: str) -> dict | None:
    return (
        (_PHASE_2C.get("cells", {}).get(f"{rung}_{lane}") or {})
        .get("ratio_fwd_over_rev", {})
        .get("batched")
    )


def _run_cell(rung: str, lane: str) -> dict:
    label = f"{rung}_{lane}"
    lane_index = LANES.index(lane)
    rung_index = RUNG_NAMES_2C.index(rung)
    model = _ladder_model(rung, solved=lane == "solved")
    register_model_pytrees(model)
    n_params = int(model.prior_count)
    print(f"\n--- {label}: n_params {n_params} ---")
    seed = SEED + 100 * rung_index + lane_index
    stream = _vector_stream(model, N_STREAM, seed)
    batch_stream = _batches(stream, N_STREAM, seed + 7)
    example = jnp.asarray(batch_stream[0])

    captures = {mode: _capture_vmapped(model, lane, mode, label) for mode in MODES}
    compiled = {mode: _compile(mode, captures[mode]["vmapped"], example, label) for mode in MODES}
    hashes = {mode: compiled[mode]["record"]["stablehlo_sha256"] for mode in MODES}
    base = {
        "rung": rung,
        "lane": lane,
        "components": RUNG_COMPONENTS[rung],
        "n_params": n_params,
        "capture": {
            mode: {k: v for k, v in captures[mode].items() if k != "vmapped"} for mode in MODES
        },
        "compile": {mode: compiled[mode]["record"] for mode in MODES},
    }
    capture_ok = (
        captures["forward"]["resolved"] == "forward"
        and captures["reverse"]["resolved"] == "reverse"
        and any(
            "forward (declared by AnalysisPoint)" in s for s in captures["forward"]["log_lines"]
        )
        and any("reverse (search override)" in s for s in captures["reverse"]["log_lines"])
    )
    executables = {mode: compiled[mode]["executable"] for mode in MODES}
    gate = _gate(executables, _gate_batches(model), hashes)
    gate["capture_mode_resolution_ok"] = bool(capture_ok)
    if not capture_ok:
        gate["pass"] = False
        gate["reasons"].append("captured search did not resolve/log the expected mode")
    base["gate"] = gate
    print(
        f"  gate {'PASS' if gate['pass'] else 'FAIL ' + str(gate['reasons'][:3])}  "
        f"obj {gate['max_rel_err_objective_fwd_vs_rev']:.1e}  "
        f"grad {gate['max_rel_err_grad_fwd_vs_rev']:.1e}  min|g| {gate['min_grad_l2']:.3g}"
    )
    if not gate["pass"]:
        print(f"  [{label}] GATE FAILED -> not timed")
        base["timed"] = False
        return base

    callables = {f"{m}_exe": executables[m] for m in MODES}
    callables.update({f"{m}_jit": captures[m]["vmapped"] for m in MODES})
    args = [jnp.asarray(b) for b in batch_stream]
    warm_s = {r: [] for r in ROUTES}
    for route in ROUTES:
        # the *_jit routes compile here (their own jit cache), untimed
        for w in range(N_WARM):
            t0 = time.perf_counter()
            block(callables[route](args[w % N_STREAM]))
            warm_s[route].append(time.perf_counter() - t0)

    times = {r: [] for r in ROUTES}
    call_index = 0
    for r in range(N_ROUNDS):
        order = ROUTES[r % len(ROUTES) :] + ROUTES[: r % len(ROUTES)]
        start = call_index
        for route in order:
            fn = callables[route]
            for c in range(N_CALLS):
                arg = args[(start + c) % N_STREAM]
                t0 = time.perf_counter()
                block(fn(arg))
                times[route].append(time.perf_counter() - t0)
        call_index = start + N_CALLS

    stats = {r: _stats_ms(times[r]) for r in ROUTES}
    ratios = {
        kind: _ratio_boots(
            times[f"forward_{kind}"],
            times[f"reverse_{kind}"],
            BOOTSTRAP_SEED + 1000 * rung_index + 10 * lane_index + j,
        )
        for j, kind in enumerate(KINDS)
    }
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
            "ratio_forward_over_reverse": ratios,
            "phase_2c_batched_ratio_same_host": _phase_2c_batched(rung, lane),
        }
    )
    p2c = base["phase_2c_batched_ratio_same_host"]
    print(
        "  median ms/step  "
        + "  ".join(f"{r} {stats[r]['median_ms']:.4f}" for r in ROUTES)
        + f"  | fwd/rev exe {ratios['exe']['ratio']:.3f} "
        f"[{ratios['exe']['ci90_low']:.3f}, {ratios['exe']['ci90_high']:.3f}]"
        f"  jit {ratios['jit']['ratio']:.3f} "
        f"[{ratios['jit']['ci90_low']:.3f}, {ratios['jit']['ci90_high']:.3f}]"
        + (f"  | 2c batched {p2c['ratio']:.3f}" if p2c else "")
    )
    return base


# ---------------------------------------------------------------------------
# End-to-end MultiStartAdam fits (L5, per lane, per mode)
# ---------------------------------------------------------------------------


def _one_fit(model, lane: str, mode: str, name: str) -> dict:
    analysis = _analysis(lane)
    search = _search(name, mode, n_steps=FIT_STEPS, seed=FIT_SEED)
    inner = search._fit
    inner_wall: dict = {}

    def timed_fit(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return inner(*args, **kwargs)
        finally:
            inner_wall["s"] = time.perf_counter() - t0

    search._fit = timed_fit
    n_log_before = len(_MODE_LOG.lines)
    t0 = time.perf_counter()
    result = search.fit(model=model, analysis=analysis)
    wall = time.perf_counter() - t0
    best = result.samples.max_log_likelihood_sample
    vector = [float(v) for v in result.samples.max_log_likelihood(as_instance=False)]
    return {
        "mode_requested": mode,
        "resolved": search._resolved_gradient_mode(analysis),
        "log_lines": _MODE_LOG.lines[n_log_before:],
        "search_fit_wall_s": float(wall),
        "search__fit_wall_s": float(inner_wall.get("s", float("nan"))),
        "best_vector": vector,
        "max_log_likelihood": float(best.log_likelihood),
        "max_log_posterior": float(best.log_posterior),
    }


def _end_to_end() -> dict:
    out = {
        "rung": "L5",
        "n_starts": BATCH,
        "n_steps": FIT_STEPS,
        "seed": FIT_SEED,
        "tolerances": {
            "best_vector_rtol": FIT_VECTOR_RTOL,
            "best_vector_atol": FIT_VECTOR_ATOL,
            "max_log_likelihood_rtol": FIT_LOG_L_RTOL,
        },
        "lanes": {},
    }
    warm_model = _ladder_model("L5", solved=True)
    register_model_pytrees(warm_model)
    t0 = time.perf_counter()
    _one_fit(warm_model, "solved", "reverse", "e2e_warmup")
    out["warmup_fit_wall_s"] = float(time.perf_counter() - t0)
    for lane in LANES:
        model = _ladder_model("L5", solved=lane == "solved")
        register_model_pytrees(model)
        fits = {mode: _one_fit(model, lane, mode, f"e2e_L5_{lane}_{mode}") for mode in MODES}
        fv = np.asarray(fits["forward"]["best_vector"])
        rv = np.asarray(fits["reverse"]["best_vector"])
        vec_rel = _rel(fv, rv)
        vec_ok = bool(np.all(np.abs(fv - rv) <= FIT_VECTOR_ATOL + FIT_VECTOR_RTOL * np.abs(rv)))
        ll_rel = _rel(fits["forward"]["max_log_likelihood"], fits["reverse"]["max_log_likelihood"])
        ll_ok = ll_rel <= FIT_LOG_L_RTOL
        mode_ok = (
            fits["forward"]["resolved"] == "forward"
            and fits["reverse"]["resolved"] == "reverse"
            and any(
                "forward (declared by AnalysisPoint)" in s for s in fits["forward"]["log_lines"]
            )
            and any("reverse (search override)" in s for s in fits["reverse"]["log_lines"])
        )
        out["lanes"][lane] = {
            "fits": fits,
            "best_vector_max_rel_err": vec_rel,
            "max_log_likelihood_rel_err": ll_rel,
            "pass": bool(vec_ok and ll_ok and mode_ok),
            "mode_resolution_ok": bool(mode_ok),
            "wall_ratio_forward_over_reverse": {
                "search_fit": fits["forward"]["search_fit_wall_s"]
                / fits["reverse"]["search_fit_wall_s"],
                "search__fit": fits["forward"]["search__fit_wall_s"]
                / fits["reverse"]["search__fit_wall_s"],
            },
        }
        print(
            f"  e2e L5_{lane}: {'PASS' if out['lanes'][lane]['pass'] else 'FAIL'}  "
            f"vec rel {vec_rel:.1e}  logL rel {ll_rel:.1e}  "
            f"_fit wall fwd {fits['forward']['search__fit_wall_s']:.2f} s  "
            f"rev {fits['reverse']['search__fit_wall_s']:.2f} s  "
            f"(fit() {fits['forward']['search_fit_wall_s']:.2f} / "
            f"{fits['reverse']['search_fit_wall_s']:.2f} s)"
        )
    out["pass"] = all(v["pass"] for v in out["lanes"].values())
    return out


# ===========================================================================
# Run
# ===========================================================================

config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_device_fp64"
)
_phase_2c_path = (
    _ROOT
    / "results"
    / "breakdown"
    / "point_source_source"
    / f"gradient_mode_crossover_{config_name}.json"
)
_PHASE_2C = json.loads(_phase_2c_path.read_text()) if _phase_2c_path.exists() else {}

# Declared default, checked before anything is compiled.
_probe_analysis = _analysis("solved")
declared_default = {
    "AnalysisPoint.gradient_mode": al.AnalysisPoint.gradient_mode,
    "resolve_gradient_mode(AnalysisPoint)": resolve_gradient_mode(_probe_analysis),
    "MultiStartAdam()._resolved_gradient_mode": af.MultiStartAdam()._resolved_gradient_mode(
        _probe_analysis
    ),
    "MultiStartAdam(gradient_mode='reverse')._resolved_gradient_mode": af.MultiStartAdam(
        gradient_mode="reverse"
    )._resolved_gradient_mode(_probe_analysis),
}
declared_default["pass"] = (
    declared_default["AnalysisPoint.gradient_mode"] == "forward"
    and declared_default["resolve_gradient_mode(AnalysisPoint)"] == "forward"
    and declared_default["MultiStartAdam()._resolved_gradient_mode"] == "forward"
    and declared_default["MultiStartAdam(gradient_mode='reverse')._resolved_gradient_mode"]
    == "reverse"
)
print(f"declared default: {declared_default}")

cells: dict[str, dict] = {}
try:
    for rung in RUNGS:
        for lane in LANES:
            cells[f"{rung}_{lane}"] = _run_cell(rung, lane)
    end_to_end = _end_to_end() if RUN_FITS and "L5" in RUNGS else {"skipped": True}
finally:
    shutil.rmtree(_SEARCH_OUTPUT, ignore_errors=True)

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

all_gates = bool(declared_default["pass"]) and all(c["gate"]["pass"] for c in cells.values())
if isinstance(end_to_end, dict) and "pass" in end_to_end:
    all_gates = all_gates and bool(end_to_end["pass"])

summary = {
    "cell": "gradient_mode_library_ab",
    "issue": "PyAutoLabs/autolens_profiling#334",
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
    "library_files": {m.__name__: m.__file__ for m in (af, aa, al)},
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
        "real_path": "search.fit(model, analysis) with jax.jit wrapped by a recorder that keeps "
        "the jitted _value_and_grad_finite (= MultiStartGradient._fit's _vmapped = "
        "jax.jit(jax.vmap(_value_and_grad_finite)) over value_and_grad_from(fitness.call, mode)) "
        "and aborts the fit before any compile; nothing re-implemented",
        "objective": "Fitness.call with fom_is_log_likelihood=False, convert_to_chi_squared=True "
        "(-2 log posterior); outputs (fom, grad, all(isfinite(grad)), constraint violation)",
        "modes": {
            "forward": "af.MultiStartAdam() with no override (AnalysisPoint's declaration)",
            "reverse": "af.MultiStartAdam(gradient_mode='reverse')",
        },
        "routes": {
            "exe": "_vmapped.lower(example).compile() executable (comparable to phase 2c batched)",
            "jit": "the captured jitted _vmapped called directly (the step loop's dispatch)",
        },
        "batch": f"n_starts = B = {BATCH}, batch_size=None (one vmap over all starts)",
        "lanes": {
            "solved": "PointSolved + FitPositionsSourceSolved",
            "plain": "PointFlux + FitPositionsSource (free source centre + flux, +3 params)",
        },
        "rungs": list(RUNGS),
        "n_rounds": N_ROUNDS,
        "n_calls_per_round": N_CALLS,
        "n_warm": N_WARM,
        "n_stream": N_STREAM,
        "instance_seed": SEED,
        "instance_stream": "phase 2c's: 16 fixed-seed batches of B draws from a 16-vector stream "
        "(entry 0 prior medians, rest vector_from_unit_vector(U(0.25,0.75))), same per-rung seeds",
        "route_order": "4 routes round-robin, start rotated each round",
        "gate_draws": f"one B-batch per PRNGKey 0..{N_GATE_KEYS - 1}, U(0.25,0.75) unit cube",
        "gate_tolerances": {
            "objective_rtol": VALUE_RTOL,
            "grad_rtol": GRAD_RTOL,
            "grad_atol": f"{GRAD_ATOL_SCALE} x max|grad_reverse| per row",
        },
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": "90%"},
        "cache_trap_guard": "fresh AnalysisPoint + search per capture; jax.clear_caches() before "
        "each lower; forward/reverse StableHLO hashes distinct",
        "phase_2c_reference": str(_phase_2c_path.relative_to(_ROOT))
        if _PHASE_2C
        else "not found for this config_name",
        "solver": "solver=None; FitPositionsSource never invokes a PointSolver",
        "pytrees": "autofit.jax.register_model(model) for every model before any trace",
    },
    "declared_default": declared_default,
    "all_gates_pass": bool(all_gates),
    "cells": cells,
    "end_to_end_fits": end_to_end,
    "wall_s": float(time.perf_counter() - _WALL_START),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_source",
    default_basename="gradient_mode_library_ab_local",
    cell="gradient_mode_library_ab",
)
dict_path.write_text(json.dumps(_scrub(summary), indent=2))

# ---------------------------------------------------------------------------
# PNG: forward/reverse ratio per rung x lane (exe, jit) vs phase 2c's batched row
# ---------------------------------------------------------------------------
timed = [c for c in cells.values() if c.get("timed")]
fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
x = np.arange(len(timed))
width = 0.26
series = (
    ("exe", "library _vmapped (AOT exe)", "#4C72B0"),
    ("jit", "library _vmapped (jit dispatch)", "#55A868"),
)
for j, (kind, text, colour) in enumerate(series):
    r = [c["ratio_forward_over_reverse"][kind]["ratio"] for c in timed]
    lo = [c["ratio_forward_over_reverse"][kind]["ci90_low"] for c in timed]
    hi = [c["ratio_forward_over_reverse"][kind]["ci90_high"] for c in timed]
    yerr = [np.subtract(r, lo), np.subtract(hi, r)]
    ax.bar(x + (j - 1) * width, r, width, yerr=yerr, capsize=3, color=colour, label=text)
p2c = [c.get("phase_2c_batched_ratio_same_host") for c in timed]
if any(p2c):
    r = [p["ratio"] if p else np.nan for p in p2c]
    lo = [p["ci90_low"] if p else np.nan for p in p2c]
    hi = [p["ci90_high"] if p else np.nan for p in p2c]
    yerr = [np.subtract(r, lo), np.subtract(hi, r)]
    ax.bar(x + width, r, width, yerr=yerr, capsize=3, color="#C44E52", label="phase 2c batched")
ax.axhline(1.0, color="k", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f"{c['rung']} {c['lane']}\n(n={c['n_params']})" for c in timed])
ax.set_ylabel("median time ratio forward / reverse (90% CI)")
ax.legend(fontsize=8)
fig.suptitle(
    f"MultiStartAdam batched step via gradient_mode, B={BATCH} ({config_name}); "
    f"rounds={N_ROUNDS}, calls/round={N_CALLS}",
    fontsize=10,
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 100)
print(f"GRADIENT_MODE LIBRARY A/B  ({config_name}, rounds={N_ROUNDS}, calls/round={N_CALLS})")
print("=" * 100)
print(
    f"  {'cell':<12} {'n':>3} {'gate':>5} {'fwd_exe':>9} {'rev_exe':>9} {'ratio':>7} "
    f"{'fwd_jit':>9} {'rev_jit':>9} {'ratio':>7} {'2c':>7} {'cmp fwd/rev s':>14}"
)
for label, c in cells.items():
    if not c.get("timed"):
        print(f"  {label:<12} {c['n_params']:>3}  FAIL (not timed)")
        continue
    s, rr = c["stats_per_call"], c["ratio_forward_over_reverse"]
    p = c.get("phase_2c_batched_ratio_same_host")
    print(
        f"  {label:<12} {c['n_params']:>3} {'ok':>5} {s['forward_exe']['median_ms']:9.4f} "
        f"{s['reverse_exe']['median_ms']:9.4f} {rr['exe']['ratio']:7.3f} "
        f"{s['forward_jit']['median_ms']:9.4f} {s['reverse_jit']['median_ms']:9.4f} "
        f"{rr['jit']['ratio']:7.3f} {(p['ratio'] if p else float('nan')):7.3f} "
        f"{c['compile']['forward']['lower_plus_compile_s']:6.2f}/"
        f"{c['compile']['reverse']['lower_plus_compile_s']:<6.2f}"
    )
print(f"  declared default pass: {declared_default['pass']}   all gates pass: {all_gates}")
print(f"  wall: {summary['wall_s']:.0f} s")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
if not all_gates:
    sys.exit(1)
