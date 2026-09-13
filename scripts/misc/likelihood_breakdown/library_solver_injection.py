"""Harness-level solver injection into the library's positive-only solve.

Why this module exists
----------------------

Phase 0 of the ``fixed-lens-light-profiling`` epic
(``results/notes/fixed_lens_light_source_only_2026_09.md``, autolens_profiling#248)
measured the **certified active-set** positivity solve as a *kernel*: 4.2 ms on
Delaunay (pass 2) and 11.0 ms on rectangular (pass 7), against the 26-28 ms S3
PDIP row. What it could not measure is the row production actually pays — the
whole ``FitImaging`` likelihood with that solver inside it. Substituting one row
for another gives ``38.4 - (25.8 - 11.0) ≈ 23.7 ms``, and the phase-0 note is
explicit that this is *arithmetic on measured rows*, not a measurement.

The library has no active-set solver, and this task does not add one: **nothing
in PyAutoArray moves.** Instead the cell patches the library's own positive-only
entry point for the duration of a timed route, so that every other part of the
likelihood — the mapper, the ``F + λH`` assembly, both log determinants, the
evidence — is still the library's own code on the library's own path. The
harness calls the library, not a copy of it.

Where the patch goes, and why it takes
--------------------------------------

``autoarray/inversion/inversion/abstract.py`` imports the *module*
(``from autoarray.inversion.inversion import inversion_util``, line 22) and
resolves ``inversion_util.reconstruction_positive_only_from`` **at call time**
(lines 619 and 645). Rebinding that module attribute is therefore picked up by
the next call; nothing binds the function at import.

What the wrapper receives
-------------------------

Under edge zeroing (``Settings.use_edge_zeroed_pixels``, the shipped default)
``Inversion.reconstruction`` does **not** hand the solver the full system: it
subsets ``curvature_reg_matrix`` and ``data_vector`` to ``solve_ids_to_keep``
first (abstract.py:607-618) and scatters exact zeros back afterwards. So the
wrapper's problem is already edge-zeroed, and its ``fixed0`` mask is
``zeros(n_keep)`` — the border pixels are not variables of the QP it sees.

That is *the same problem* phase 0 certified. The phase-0 kernel ran at full
size with the edge-zero indices seeded into the fixed set and never released
(``active_set_masked_jax``'s ``permanent`` mask, which ``freeable = Z & ~permanent``
excludes from every release), so its free-block solves and its KKT tests are
index-for-index the ones taken here on the subset. The certifying pass budgets
phase 0 measured therefore transfer, and the cell records the certification
diagnostics per budget rather than assuming they do.

Jacobi scaling
--------------

The wrapper replaces the whole library function, so it must reproduce the
library's own preconditioning (``inversion_util.py:355-375``): ``d = sqrt(diag Q)``,
``D = 1/d``, solve ``(D Q D) y = D q``, return ``y * D``. That is exactly
``reconstruction_steps.jacobi_scaled``, which is why the phase-0 kernels and the
library agree to 1e-10 nats on the same system.

``lax.cond`` under ``vmap``
---------------------------

The production shape of the certified route is "N-pass budget, PDIP fallback
when the budget is exhausted", which is a ``jax.lax.cond``. Under ``jax.jit``
that executes one branch. Under ``jax.vmap`` a batched predicate turns ``cond``
into ``select``, which evaluates **both** — so a batched certified-with-fallback
row costs the certified solve *plus* the PDIP solve and is not the certified
cost. :func:`certified_solver_injected` therefore takes ``fallback=False`` as
well, and the cell measures both shapes and says which is which.
"""

from __future__ import annotations

import contextlib

import numpy as np

from likelihood_breakdown import active_set_steps

__all__ = [
    "LIBRARY_POSITIVE_ONLY_DOTTED",
    "LIBRARY_POSITIVE_NEGATIVE_DOTTED",
    "certified_solver_injected",
    "positive_negative_probe",
]

#: The dotted names this module rebinds. Recorded verbatim into the results
#: JSON so a reader of the numbers knows exactly what was patched.
LIBRARY_POSITIVE_ONLY_DOTTED = (
    "autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from"
)
LIBRARY_POSITIVE_NEGATIVE_DOTTED = (
    "autoarray.inversion.inversion.inversion_util.reconstruction_positive_negative_from"
)


def certified_reconstruction_from(
    data_vector,
    curvature_reg_matrix,
    *,
    n_passes: int,
    fallback: bool,
    original,
    tau_rel: float = active_set_steps.TAU_REL_DEFAULT,
    settings=None,
    xp=np,
    fingerprint=None,
):
    """The certified active-set solve of ``(curvature_reg_matrix, data_vector)``.

    A drop-in for ``inversion_util.reconstruction_positive_only_from`` on the
    JAX path. ``n_passes`` is the fixed pass budget (static — it is a
    ``lax.scan`` length). With ``fallback=True`` an uncertified iterate is
    discarded and *original* — the library's own PDIP — is returned instead, via
    ``lax.cond``; with ``fallback=False`` the (possibly uncertified) active-set
    iterate is returned unconditionally, which is the shape a batched row must
    use (see the module docstring).
    """
    import jax
    import jax.numpy as jnp

    d = jnp.sqrt(jnp.diag(curvature_reg_matrix))
    scale = 1.0 / d
    Q_pc = (curvature_reg_matrix * scale[:, None]) * scale[None, :]
    q_pc = data_vector * scale

    n = q_pc.shape[0]
    out = active_set_steps.active_set_masked_jax(
        Q_pc, q_pc, jnp.zeros(n, dtype=bool), int(n_passes), tau_rel=tau_rel
    )
    x_active = out["x"] * scale

    if not fallback:
        return x_active

    certified = jnp.any(out["certified"])

    def _library_pdip():
        return original(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            settings=settings,
            xp=xp,
            fingerprint=fingerprint,
        )

    return jax.lax.cond(certified, lambda: x_active, _library_pdip)


@contextlib.contextmanager
def certified_solver_injected(
    n_passes: int,
    *,
    fallback: bool = True,
    tau_rel: float = active_set_steps.TAU_REL_DEFAULT,
):
    """Rebind the library's positive-only solve to the certified scheme.

    Scoped: the original function is restored on exit, including on an
    exception. Only the **JAX** path is redirected — a NumPy-path call (an eager
    ``FitImaging`` the cell builds for its diagnostics, say) is delegated to the
    original untouched, so a patch left open around eager work cannot silently
    change a reference number.

    ``tau_rel`` is the active set's relative KKT tolerance. It is a
    **parameter, not a constant**, because a tolerance calibrated in fp64 is
    not a tolerance in fp32: under ``Settings.use_mixed_precision`` the library
    accumulates the curvature matrix in float32, and certifying at the fp64
    default would certify against the matrix's own round-off. The caller
    re-derives it per precision and records the value it used.

    Yields a mutable dict of call counters (``{"jax": n, "numpy": n}``) so the
    caller can assert the patched path was actually taken — a patch that never
    fires would otherwise report the library's own PDIP timing as the certified
    route's.
    """
    from autoarray.inversion.inversion import inversion_util

    original = inversion_util.reconstruction_positive_only_from
    counts = {"jax": 0, "numpy": 0}

    def patched(data_vector, curvature_reg_matrix, settings=None, xp=np, fingerprint=None):
        if not xp.__name__.startswith("jax"):
            counts["numpy"] += 1
            return original(
                data_vector=data_vector,
                curvature_reg_matrix=curvature_reg_matrix,
                settings=settings,
                xp=xp,
                fingerprint=fingerprint,
            )
        counts["jax"] += 1
        return certified_reconstruction_from(
            data_vector,
            curvature_reg_matrix,
            n_passes=n_passes,
            fallback=fallback,
            original=original,
            tau_rel=tau_rel,
            settings=settings,
            xp=xp,
            fingerprint=fingerprint,
        )

    inversion_util.reconstruction_positive_only_from = patched
    try:
        yield counts
    finally:
        inversion_util.reconstruction_positive_only_from = original


@contextlib.contextmanager
def positive_negative_probe():
    """Count calls into the library's positive-**negative** solve, changing nothing.

    ``Settings(use_positive_only_solver=False)`` is supposed to route the
    reconstruction to ``reconstruction_positive_negative_from`` — one
    ``xp.linalg.solve`` — with edge zeroing silently off (``solve_ids_to_keep``
    returns ``None`` the moment positivity is off, abstract.py:540). This probe
    is how the cell *verifies* that rather than asserting it in prose: the
    counter must be non-zero on a route-(c) call and the positive-only counter
    must be zero.
    """
    from autoarray.inversion.inversion import inversion_util

    original = inversion_util.reconstruction_positive_negative_from
    counts = {"calls": 0}

    def patched(data_vector, curvature_reg_matrix, xp=np):
        counts["calls"] += 1
        return original(data_vector=data_vector, curvature_reg_matrix=curvature_reg_matrix, xp=xp)

    inversion_util.reconstruction_positive_negative_from = patched
    try:
        yield counts
    finally:
        inversion_util.reconstruction_positive_negative_from = original
