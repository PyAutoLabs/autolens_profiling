"""The candidate positive-only solvers of the linear-solver programme.

Every candidate is a *composition of library primitives* — ``autoarray.util.jax_nnls``,
``autoarray.util.jax_active_set`` and ``autoarray.util.fnnls`` — never a transcription of their
internals, so a row in a study is a statement about the library's own solver code at the
installed revision. Where a candidate reproduces a production path it builds the Jacobi
quantities exactly as ``inversion_util.reconstruction_positive_only_from`` does::

    d = sqrt(diag(Q)); D = 1 / d; Q_pc = Q * D[:, None] * D[None, :]; q_pc = q * D

and returns the reconstruction in raw coordinates the way that function does (``y * D`` or
``x * D``).

Registry contract
-----------------

``CANDIDATES[name]`` is a :class:`Candidate`. ``candidate.fn(Q, q, meta, *, target_kappa=1e-11)``
returns ``(x, stats)`` with ``x`` a float64 NumPy array in raw coordinates and ``stats`` a dict
that always has ``converged`` (bool or None) and ``iterations`` (int or None) plus
solver-specific extras. ``candidate.kernel(Q, q, *, target_kappa=1e-11)`` runs only the timed
computation (the jitted forward solve for JAX candidates, blocked until ready) — hand it to
``_metrics.timing`` for warm wall-clock. Each JAX candidate's kernel is ``jax.jit``-compiled
once per ``target_kappa`` (and, via jit's own cache, once per system shape).

``target_kappa`` defaults to 1e-11, the library's ``general.inversion.nnls_target_kappa``. It
only affects the relaxed-KKT *backward* preparation, never a forward value, so here it matters
for the ``pdip_raw`` backward-status extras alone.

Adding a candidate: write a builder below from library calls only, register it in
:func:`_build_registry`, document the composed library function in its description, and add
its row to the candidate table in ``scripts/lens/solver/README.md``.

Post-hoc (exploratory) candidates
---------------------------------

:data:`POSTHOC_CANDIDATES` are **not** part of the phase-1 pre-registered rule
(``results/notes/linear_solver_accuracy_2026_09.md``, commit ``327f571``). They were added after
the deciding run as *inputs to the phase-2 fix*, not as contenders for its verdict: the polish
with a larger cap (``pdip_raw_polish_cap{20,50}``), jaxnnls's tight absolute tolerance with a
larger iteration cap (``pdip_raw_tol_jaxnnls_cap{100,200}``) and the replace-the-constant
tolerance family tightened further (``pdip_raw_tol_1e-4/-5/-6``). ``accuracy.py --posthoc`` runs
them into a separate ``accuracy_posthoc_summary_*`` artefact so the pre-registered table never
mixes with them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache

import numpy as np

DEFAULT_TARGET_KAPPA = 1.0e-11
RAW_CAPS = (8, 12, 16, 24, 32, 50, 100, 200)
TOL_FACTORS = {"1e-1": 1.0e-1, "1e-2": 1.0e-2, "1e-3": 1.0e-3}
#: Post-hoc (exploratory, not pre-registered) extensions — see the module docstring.
POSTHOC_TOL_FACTORS = {"1e-4": 1.0e-4, "1e-5": 1.0e-5, "1e-6": 1.0e-6}
POSTHOC_POLISH_CAPS = (20, 50)
POSTHOC_JAXNNLS_CAPS = (100, 200)


@dataclass(frozen=True)
class Candidate:
    """One registered solver: name, prose description, the library function(s) it composes."""

    name: str
    description: str
    library_function: str
    backend: str  # "jax" | "numpy"
    fn: Callable
    kernel: Callable
    #: JAX candidates only: ``build(target_kappa)`` -> the pure ``(Q, q) -> dict`` body that
    #: ``kernel`` jits; :func:`batched_kernel` vmaps it. ``None`` for NumPy candidates.
    build: Callable | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


def _jacobi(Q, q):
    """The Jacobi quantities exactly as ``reconstruction_positive_only_from`` builds them."""
    import jax.numpy as jnp

    d = jnp.sqrt(jnp.diag(Q))
    D = 1.0 / d
    Q_pc = (Q * D[:, None]) * D[None, :]
    q_pc = q * D
    return Q_pc, q_pc, D


def _to_device(Q, q):
    import jax.numpy as jnp

    return jnp.asarray(Q, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)


def _as_int(v):
    return None if v is None else int(np.asarray(v))


def _jax_candidate(name, description, library_function, build, extras=None) -> Candidate:
    """Wrap a jitted kernel builder into a :class:`Candidate`.

    ``build(target_kappa)`` returns a pure ``(Q, q) -> dict`` function with at least ``x``,
    ``converged`` and ``iterations``; it is ``jax.jit``-ed once per ``target_kappa``.
    ``extras(Q, q, target_kappa)`` optionally returns diagnostic arrays computed outside the
    timed kernel.
    """

    @cache
    def jitted(target_kappa):
        return _jax().jit(build(target_kappa))

    def kernel(Q, q, *, target_kappa=DEFAULT_TARGET_KAPPA):
        Qj, qj = _to_device(Q, q)
        return _jax().block_until_ready(jitted(float(target_kappa))(Qj, qj))

    def fn(Q, q, meta=None, *, target_kappa=DEFAULT_TARGET_KAPPA):
        out = kernel(Q, q, target_kappa=target_kappa)
        x = np.asarray(out["x"], dtype=np.float64)
        stats = {
            "converged": bool(np.asarray(out["converged"])),
            "iterations": _as_int(out["iterations"]),
        }
        for key, value in out.items():
            if key not in ("x", "converged", "iterations"):
                stats[key] = _as_int(value)
        if extras is not None:
            stats.update(extras(Q, q, float(target_kappa)))
        return x, stats

    return Candidate(name, description, library_function, "jax", fn, kernel, build)


# ---------------------------------------------------------------------------
# Builders (library compositions only)
# ---------------------------------------------------------------------------


def _build_pdip_jacobi(target_kappa):
    from autoarray.util.jax_nnls import solve_nnls_primal_with_status

    def f(Q, q):
        Q_pc, q_pc, D = _jacobi(Q, q)
        x, converged, iterations = solve_nnls_primal_with_status(
            Q_pc, q_pc, target_kappa=target_kappa, solver_tol=None, max_iter=50
        )
        return {"x": x * D, "converged": converged, "iterations": iterations}

    return f


def _build_pdip_raw(max_iter):
    def build(target_kappa):
        from autoarray.util.jax_nnls import solve_nnls_primal_raw_forward

        def f(Q, q):
            Q_pc, q_pc, D = _jacobi(Q, q)
            y, converged, iterations = solve_nnls_primal_raw_forward(
                Q_pc, q_pc, Q, q, D, target_kappa=target_kappa, solver_tol=None, max_iter=max_iter
            )
            return {"x": y * D, "converged": converged, "iterations": iterations}

        return f

    return build


@cache
def _raw_backward_status_fn(target_kappa, max_iter):
    from autoarray.util.jax_nnls import raw_forward_backward_status

    def f(Q, q):
        Q_pc, q_pc, D = _jacobi(Q, q)
        return raw_forward_backward_status(
            Q_pc, q_pc, Q, q, D, target_kappa=target_kappa, solver_tol=None, max_iter=max_iter
        )

    return _jax().jit(f)


def _raw_backward_extras(max_iter):
    def extras(Q, q, target_kappa):
        Qj, qj = _to_device(Q, q)
        rc, ri, pc, pi = _raw_backward_status_fn(target_kappa, max_iter)(Qj, qj)
        return {
            "backward_relaxed_converged": _as_int(rc),
            "backward_relaxed_iter": _as_int(ri),
            "backward_polish_converged": _as_int(pc),
            "backward_polish_iter": _as_int(pi),
        }

    return extras


def _build_pdip_raw_tol(factor):
    def build(target_kappa):
        from autoarray.util.jax_nnls import (
            DATA_SCALED_TOL_FACTOR,
            data_scaled_solver_tol,
            solve_nnls,
        )

        def f(Q, q):
            _, _, D = _jacobi(Q, q)
            tol = data_scaled_solver_tol(q) * (factor / DATA_SCALED_TOL_FACTOR)
            x, _, _, converged, iterations = solve_nnls(Q, q, solver_tol=tol, max_iter=50)
            return {"x": (x / D) * D, "converged": converged, "iterations": iterations}

        return f

    return build


def _build_pdip_raw_tol_jaxnnls(max_iter=50):
    def build(target_kappa):
        from autoarray.util.jax_nnls import solve_nnls

        def f(Q, q):
            _, _, D = _jacobi(Q, q)
            x, _, _, converged, iterations = solve_nnls(Q, q, solver_tol=None, max_iter=max_iter)
            return {"x": (x / D) * D, "converged": converged, "iterations": iterations}

        return f

    return build


def _build_pdip_raw_polish(polish_max_iter=None):
    """Forward raw PDIP + the #573 backward polish rule; ``None`` = the library constant."""

    def build(target_kappa):
        return _pdip_raw_polish_fn(polish_max_iter)

    return build


def _pdip_raw_polish_fn(polish_max_iter):
    from autoarray.util.jax_nnls import (
        RAW_BACKWARD_POLISH_MAX_ITER,
        data_scaled_solver_tol,
        solve_nnls,
    )

    polish_cap = RAW_BACKWARD_POLISH_MAX_ITER if polish_max_iter is None else int(polish_max_iter)

    def f(Q, q):
        import jax.numpy as jnp

        Q_pc, q_pc, D = _jacobi(Q, q)
        x, s, z, converged, iterations = solve_nnls(
            Q, q, solver_tol=data_scaled_solver_tol(q), max_iter=50
        )
        y, sy, zy = x / D, s / D, z * D
        yp, sp, zp, polish_converged, polish_iter = solve_nnls(
            Q_pc, q_pc, max_iter=polish_cap, init=(y, sy, zy)
        )
        ok = jnp.logical_and(
            polish_converged == 1,
            jnp.all(jnp.isfinite(yp)) & jnp.all(sp > 0) & jnp.all(zp > 0),
        )
        y_out = jnp.where(ok, yp, y)
        return {
            "x": y_out * D,
            "converged": converged,
            "iterations": iterations + polish_iter,
            "forward_iterations": iterations,
            "polish_converged": polish_converged,
            "polish_iter": polish_iter,
            "polish_kept": ok.astype(int),
        }

    return f


def _build_certified(target_kappa):
    import autoarray as aa
    from autoarray.util.jax_active_set import solve_certified

    settings = aa.Settings()
    pass_budget = int(settings.certified_pass_budget)
    tau_rel = float(settings.certified_tau_rel)

    def f(Q, q):
        Q_pc, q_pc, D = _jacobi(Q, q)
        x, certified, passes = solve_certified(Q_pc, q_pc, pass_budget=pass_budget, tau_rel=tau_rel)
        return {"x": x * D, "converged": certified, "iterations": passes, "certified": certified}

    return f


# --- NumPy reference ---------------------------------------------------------


def _fnnls_kernel(Q, q, *, target_kappa=DEFAULT_TARGET_KAPPA):
    from autoarray.util.fnnls import fnnls_cholesky

    Q = np.asarray(Q, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    stats: dict = {}
    x = fnnls_cholesky(Q, q, P_initial=np.linalg.solve(Q, q) > 0, stats=stats)
    return np.asarray(x, dtype=np.float64), stats


def _fnnls_fn(Q, q, meta=None, *, target_kappa=DEFAULT_TARGET_KAPPA):
    x, st = _fnnls_kernel(Q, q)
    return x, {
        "converged": True,
        "iterations": _as_int(st.get("outer_iterations")),
        "inner_iterations": _as_int(st.get("inner_iterations")),
        "n_passive": _as_int(st.get("n_passive")),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, Candidate]:
    reg: dict[str, Candidate] = {}

    def add(c: Candidate):
        reg[c.name] = c

    add(
        Candidate(
            "fnnls",
            "Reference. NumPy fnnls with incremental Cholesky from the dense-sign start "
            "(P_initial = solve(Q, q) > 0), the NumPy path of reconstruction_positive_only_from "
            "with the warm-start memo off. Defines x_ref; its metrics are a self-check.",
            "autoarray.util.fnnls.fnnls_cholesky",
            "numpy",
            _fnnls_fn,
            _fnnls_kernel,
        )
    )
    add(
        _jax_candidate(
            "pdip_jacobi",
            "Jacobi-preconditioned PDIP: solve (D Q D) y = D q at jaxnnls's own tolerance "
            "min(n eps 5e3, 1e-2), cap 50, return y * D — the library 'jacobi' mode (the default "
            "for inversions with a Mapper; the MGE default before PyAutoArray#571).",
            "autoarray.util.jax_nnls.solve_nnls_primal_with_status",
            _build_pdip_jacobi,
        )
    )
    add(
        _jax_candidate(
            "pdip_raw",
            "Library default for linear-object-only (MGE) inversions: forward PDIP on the raw "
            "(Q, q) at data_scaled_solver_tol(q), cap 50; the library maps the iterate to "
            "y = x / D and, since PyAutoArray#595, returns the #573 polish of it (at most "
            "RAW_POLISH_MAX_ITER tight PDIP iterations on (Q_pc, q_pc), kept iff converged and "
            "interior) — reconstruction y * D, reproduced exactly. converged / iterations are "
            "the raw forward solve's (the polish is not counted). Extras: the polish / "
            "relaxed-KKT status from raw_forward_backward_status (not timed).",
            "autoarray.util.jax_nnls.solve_nnls_primal_raw_forward (+ raw_forward_backward_status)",
            _build_pdip_raw(50),
            extras=_raw_backward_extras(50),
        )
    )
    for label, factor in TOL_FACTORS.items():
        add(
            _jax_candidate(
                f"pdip_raw_tol_{label}",
                f"Raw forward PDIP with the data-scaled tolerance at factor {label} in place of "
                f"DATA_SCALED_TOL_FACTOR (tol = {label} * n * eps * max(1, max|q|), i.e. "
                "data_scaled_solver_tol(q) * factor / DATA_SCALED_TOL_FACTOR), cap 50. The "
                "1e-2 row reproduces the released tolerance through solve_nnls directly, with "
                "no polish (so, since PyAutoArray#595, not pdip_raw's value).",
                "autoarray.util.jax_nnls.solve_nnls + data_scaled_solver_tol",
                _build_pdip_raw_tol(factor),
            )
        )
    add(
        _jax_candidate(
            "pdip_raw_tol_jaxnnls",
            "Raw forward PDIP at jaxnnls's absolute tolerance min(n eps 5e3, 1e-2) "
            "(solver_tol=None on the raw system), cap 50 — the tolerance that is unreachable on "
            "an unscaled system; expected to run to the cap.",
            "autoarray.util.jax_nnls.solve_nnls(solver_tol=None)",
            _build_pdip_raw_tol_jaxnnls(50),
        )
    )
    add(
        _jax_candidate(
            "pdip_raw_polish",
            "Raw forward PDIP (data-scaled tol, cap 50), then at most "
            "RAW_BACKWARD_POLISH_MAX_ITER tight PDIP iterations on (Q_pc, q_pc) warm-started "
            "from the mapped iterate (y, s/D, z*D); the polished y is kept iff the polish "
            "converges and is finite with s, z > 0 — PyAutoArray#573's backward-pass rule, "
            "applied here to the FORWARD value. iterations = forward + polish.",
            "autoarray.util.jax_nnls.solve_nnls (the _raw_forward_backward_point polish rule)",
            _build_pdip_raw_polish(None),
        )
    )
    for cap in RAW_CAPS:
        add(
            _jax_candidate(
                f"pdip_raw_cap_{cap}",
                f"pdip_raw (released raw mode, data-scaled tolerance) with max_iter={cap}.",
                "autoarray.util.jax_nnls.solve_nnls_primal_raw_forward",
                _build_pdip_raw(cap),
            )
        )
    add(
        _jax_candidate(
            "certified",
            "Certified active-set solve on the Jacobi system (the library 'certified' solver, "
            "used for Mapper-only inversions), pass budget / tau_rel from Settings(), NO PDIP "
            "fallback so an uncertified iterate is scored as-is (stats.certified). Context "
            "only: the library never routes MGE systems here.",
            "autoarray.util.jax_active_set.solve_certified",
            _build_certified,
        )
    )
    # --- Post-hoc (exploratory; NOT pre-registered) ---------------------------------------
    for cap in POSTHOC_POLISH_CAPS:
        add(
            _jax_candidate(
                f"pdip_raw_polish_cap{cap}",
                f"POST-HOC (exploratory, not pre-registered). pdip_raw_polish with the polish "
                f"capped at {cap} PDIP iterations instead of RAW_BACKWARD_POLISH_MAX_ITER, same "
                "keep rule (converged, finite, s, z > 0). iterations = forward + polish.",
                "autoarray.util.jax_nnls.solve_nnls (the _raw_forward_backward_point polish rule)",
                _build_pdip_raw_polish(cap),
            )
        )
    for cap in POSTHOC_JAXNNLS_CAPS:
        add(
            _jax_candidate(
                f"pdip_raw_tol_jaxnnls_cap{cap}",
                f"POST-HOC (exploratory, not pre-registered). pdip_raw_tol_jaxnnls (jaxnnls's "
                f"absolute tolerance on the raw system) with max_iter={cap} instead of 50 — what "
                "the tight tolerance costs in iterations when the cap no longer truncates it.",
                "autoarray.util.jax_nnls.solve_nnls(solver_tol=None)",
                _build_pdip_raw_tol_jaxnnls(cap),
            )
        )
    for label, factor in POSTHOC_TOL_FACTORS.items():
        add(
            _jax_candidate(
                f"pdip_raw_tol_{label}",
                f"POST-HOC (exploratory, not pre-registered). Raw forward PDIP with the "
                f"data-scaled tolerance at factor {label} in place of DATA_SCALED_TOL_FACTOR, "
                "cap 50 — the replace-the-constant family, tightened past 1e-3.",
                "autoarray.util.jax_nnls.solve_nnls + data_scaled_solver_tol",
                _build_pdip_raw_tol(factor),
            )
        )
    return reg


CANDIDATES: dict[str, Candidate] = _build_registry()

#: The candidates ``accuracy.py`` runs by default (the iteration-cap sweep is
#: ``early_stopping.py``'s job).
#: The post-hoc (exploratory, not pre-registered) candidates — ``accuracy.py --posthoc``.
POSTHOC_CANDIDATES = (
    [f"pdip_raw_polish_cap{cap}" for cap in POSTHOC_POLISH_CAPS]
    + [f"pdip_raw_tol_jaxnnls_cap{cap}" for cap in POSTHOC_JAXNNLS_CAPS]
    + [f"pdip_raw_tol_{label}" for label in POSTHOC_TOL_FACTORS]
)
#: The pre-registered phase-1 set (everything except the cap sweep and the post-hoc set).
ACCURACY_DEFAULT = [
    n for n in CANDIDATES if not n.startswith("pdip_raw_cap_") and n not in POSTHOC_CANDIDATES
]
#: ``--posthoc`` runs the released default and the pre-registered anchors beside the post-hoc
#: set, so the separate artefact reads on its own.
POSTHOC_DEFAULT = ["fnnls", "pdip_raw", "pdip_raw_polish", "pdip_raw_tol_jaxnnls"] + list(
    POSTHOC_CANDIDATES
)
#: The early-stopping sweep, in cap order.
CAP_CANDIDATES = [f"pdip_raw_cap_{cap}" for cap in RAW_CAPS]


@cache
def batched_kernel(candidate: Candidate, target_kappa: float = DEFAULT_TARGET_KAPPA):
    """``jax.jit(jax.vmap(body))`` of a JAX candidate's solve, ``target_kappa`` closed over.

    Takes stacked ``(B, n, n)`` / ``(B, n)`` device arrays and returns per-lane
    ``(x, converged, iterations)`` with leading axis ``B`` (``x`` in raw coordinates). The body
    is the same ``build(target_kappa)`` function :attr:`Candidate.kernel` jits unbatched, so a
    lane solves exactly what the unbatched cells solve; ``fn`` / ``kernel`` are unchanged. Under
    ``vmap`` a ``while_loop`` runs until its slowest lane stops, so every lane pays the batch's
    maximum iteration count in wall time (the per-lane ``iterations`` are still each lane's own).
    jit caches one compile per batch shape.
    """
    if candidate.build is None:
        raise ValueError(f"{candidate.name} is not a JAX candidate; it has no batched kernel")
    jax = _jax()
    body = candidate.build(float(target_kappa))

    def lane(Q, q):
        out = body(Q, q)
        return out["x"], out["converged"], out["iterations"]

    return jax.jit(jax.vmap(lane))


def warm_up(candidate: Candidate, Q, q, *, target_kappa=DEFAULT_TARGET_KAPPA) -> float:
    """Run ``candidate.kernel`` once (the compile for JAX candidates) and return its wall in s."""
    t0 = time.perf_counter()
    candidate.kernel(Q, q, target_kappa=target_kappa)
    return time.perf_counter() - t0
