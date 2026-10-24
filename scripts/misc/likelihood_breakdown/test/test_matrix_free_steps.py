"""Unit tests for the matrix-free likelihood kernels (``matrix_free_steps.py``).

CPU only, fp64. Two tiers:

1. **Dense references, no imaging.** The solvers take a plain ``matvec``
   callable, so PCG is checked against ``jnp.linalg.solve``, the matrix-free
   PDIP against ``autoarray.util.jax_nnls.solve_nnls`` and the SLQ log-det
   against ``jnp.linalg.slogdet`` on random SPD systems. These pin the
   *algorithms*.
2. **A tiny ``InversionImagingSparse``.** A 30x30 simulated dataset, an 8x8
   ``RectangularUniform`` mesh and a 3-Gaussian linear basis (the smallest
   thing that satisfies ``sparse_context_from``'s one-mapper-plus-one-func-list
   guard) give a 67-parameter inversion whose ``curvature_reg_matrix`` the
   matvec must reproduce column by column, and whose exact Cholesky solve and
   log-dets the matrix-free kernels must match. This pins the *operator* —
   in particular that the already-PSF-operated MGE block is not blurred twice
   and that ``add_to_diag`` lands on the no-regularization parameters.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_matrix_free_steps.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import matrix_free_steps as mfs  # noqa: E402
from likelihood_breakdown import reconstruction_steps, sparse_steps  # noqa: E402

# ---------------------------------------------------------------------------
# Dense helpers
# ---------------------------------------------------------------------------


def _spd(n: int, seed: int):
    """A well-conditioned SPD matrix ``AᵀA + n I`` and a random right-hand side."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = A.T @ A + n * np.eye(n)
    q = rng.standard_normal(n)
    return jnp.asarray(Q, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)


def _spd_with_spectrum(n: int, seed: int, eigenvalues):
    """``V diag(eigenvalues) Vᵀ`` for a random orthogonal ``V`` — a known log-det."""
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return jnp.asarray((V * eigenvalues) @ V.T, dtype=jnp.float64)


def _dense_matvec(Q):
    return lambda V: Q @ V


# ---------------------------------------------------------------------------
# 1. PCG
# ---------------------------------------------------------------------------


def test_pcg_matches_dense_solve():
    n = 200
    Q, q = _spd(n, seed=0)
    reference = jnp.linalg.solve(Q, q)

    diag = jnp.diag(Q)
    x_jacobi, iters_jacobi, res_jacobi = mfs.pcg_solve(
        _dense_matvec(Q), q, tol=1e-12, maxiter=n, diag=diag
    )
    x_plain, iters_plain, res_plain = mfs.pcg_solve(
        _dense_matvec(Q), q, tol=1e-12, maxiter=n, diag=None
    )

    for x in (x_jacobi, x_plain):
        assert np.allclose(np.asarray(x), np.asarray(reference), rtol=1e-8, atol=1e-12)

    # Both preconditioners converge inside the Krylov dimension.
    assert 0 < int(iters_jacobi) < n
    assert 0 < int(iters_plain) < n
    assert float(res_jacobi) <= 1e-12
    assert float(res_plain) <= 1e-12


def test_pcg_solve_batched_matches_column_solves():
    n, batch = 120, 4
    Q, _ = _spd(n, seed=3)
    rng = np.random.default_rng(4)
    Dmat = jnp.asarray(rng.standard_normal((n, batch)), dtype=jnp.float64)

    X, iters, residual = mfs.pcg_solve_batched(
        _dense_matvec(Q), Dmat, tol=1e-12, maxiter=n, diag=jnp.diag(Q)
    )
    reference = jnp.linalg.solve(Q, Dmat)

    assert np.allclose(np.asarray(X), np.asarray(reference), rtol=1e-8, atol=1e-12)
    assert 0 < int(iters) < n
    assert np.all(np.asarray(residual) <= 1e-12)


def test_pcg_solve_is_jittable():
    n = 60
    Q, q = _spd(n, seed=5)
    diag = jnp.diag(Q)

    @jax.jit
    def solve(q):
        return mfs.pcg_solve(_dense_matvec(Q), q, tol=1e-12, maxiter=n, diag=diag)[0]

    assert np.allclose(np.asarray(solve(q)), np.asarray(jnp.linalg.solve(Q, q)), rtol=1e-8)


# ---------------------------------------------------------------------------
# 2. Matrix-free PDIP
# ---------------------------------------------------------------------------


def test_pdip_matches_solve_nnls():
    from autoarray.util.jax_nnls import solve_nnls

    n = 80
    Q, q = _spd(n, seed=1)

    # The unconstrained solution must have negative entries, or the positivity
    # constraint is inactive and the test would pass on an unconstrained solve.
    unconstrained = np.asarray(jnp.linalg.solve(Q, q))
    assert (unconstrained < 0).sum() > n // 10

    x_ref, _, _, converged_ref, _ = solve_nnls(Q, q)

    x_mf, pdip_iter, cg_iter, converged_mf = mfs.pdip_matrix_free(
        _dense_matvec(Q), q, diag=jnp.diag(Q), cg_tol=1e-12, cg_maxiter=n
    )

    assert int(converged_ref) == 1
    assert int(converged_mf) == 1
    assert int(pdip_iter) > 1
    assert int(cg_iter) > int(pdip_iter)
    assert np.all(np.asarray(x_mf) >= -1e-12)

    max_abs_diff = float(np.max(np.abs(np.asarray(x_mf) - np.asarray(x_ref))))
    assert max_abs_diff < 1e-6, f"matrix-free PDIP differs from solve_nnls by {max_abs_diff:.3e}"


# ---------------------------------------------------------------------------
# 3. SLQ log-determinant
# ---------------------------------------------------------------------------


def test_slq_logdet_within_spread():
    n, n_probes, n_lanczos = 300, 32, 40
    eigenvalues = np.logspace(0.0, 1.0, n)
    Q = _spd_with_spectrum(n, seed=2, eigenvalues=eigenvalues)

    exact = float(jnp.linalg.slogdet(Q)[1])

    key = jax.random.PRNGKey(0)
    estimate, per_probe, n_clamped = mfs.slq_logdet(
        _dense_matvec(Q), n, n_probes=n_probes, n_lanczos=n_lanczos, key=key
    )

    assert int(n_clamped) == 0

    spread = float(np.std(np.asarray(per_probe), ddof=1)) / np.sqrt(n_probes)
    error = abs(float(estimate) - exact)
    assert error < 3.0 * spread, (
        f"SLQ log-det error {error:.3e} exceeds 3 sigma ({3.0 * spread:.3e}); exact {exact:.6f}"
    )

    # A fixed key is a bit-identical repeat — the property that makes a probe
    # sweep a smooth bias rather than jitter.
    estimate_again, per_probe_again, _ = mfs.slq_logdet(
        _dense_matvec(Q), n, n_probes=n_probes, n_lanczos=n_lanczos, key=key
    )
    assert float(estimate_again) == float(estimate)
    assert np.array_equal(np.asarray(per_probe_again), np.asarray(per_probe))

    # A different key moves the estimate (it is stochastic, not a constant).
    estimate_other, _, _ = mfs.slq_logdet(
        _dense_matvec(Q),
        n,
        n_probes=n_probes,
        n_lanczos=n_lanczos,
        key=jax.random.PRNGKey(7),
    )
    assert float(estimate_other) != float(estimate)


def test_slq_logdet_reorthogonalisation_does_not_hurt():
    """``reorth=True`` is at least as accurate as ``reorth=False``, on 3 seeds.

    Loss of orthogonality at large ``m`` duplicates Ritz values and biases the
    quadrature. The bound asserted is the weak one (no worse, up to one
    standard error of the estimator) because on a well-conditioned spectrum the
    two agree to within Hutchinson noise.
    """
    n, n_probes, n_lanczos = 200, 16, 60
    eigenvalues = np.logspace(-1.0, 2.0, n)

    for seed in (2, 11, 23):
        Q = _spd_with_spectrum(n, seed=seed, eigenvalues=eigenvalues)
        exact = float(jnp.linalg.slogdet(Q)[1])
        key = jax.random.PRNGKey(seed)

        plain, per_probe, _ = mfs.slq_logdet(
            _dense_matvec(Q), n, n_probes=n_probes, n_lanczos=n_lanczos, key=key
        )
        reorthogonalised, _, _ = mfs.slq_logdet(
            _dense_matvec(Q),
            n,
            n_probes=n_probes,
            n_lanczos=n_lanczos,
            key=key,
            reorth=True,
        )

        spread = float(np.std(np.asarray(per_probe), ddof=1)) / np.sqrt(n_probes)
        assert abs(float(reorthogonalised) - exact) <= abs(float(plain) - exact) + spread


# ---------------------------------------------------------------------------
# 4. The operator, against a tiny InversionImagingSparse
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_inversion():
    """The smallest ``InversionImagingSparse`` the sparse steps accept.

    One ``Mapper`` (8x8 ``RectangularUniform``, ``al.reg.Constant``) plus one
    ``AbstractLinearObjFuncList`` (3 linear Gaussians) — the shape
    ``sparse_context_from`` guards for — on a 30x30 simulated dataset with a
    5x5 Gaussian PSF. 67 parameters, so every column of the matvec can be
    checked against the dense matrix.
    """
    autolens = pytest.importorskip("autolens")
    al = autolens

    grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.2)
    psf = al.Convolver.from_gaussian(shape_native=(5, 5), sigma=0.3, pixel_scales=0.2)

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        noise_seed=1,
    )
    truth = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.Sersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5),
                mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            al.Galaxy(
                redshift=1.0,
                bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0, effective_radius=0.2),
            ),
        ]
    )
    dataset = simulator.via_tracer_from(tracer=truth, grid=grid)
    dataset = dataset.apply_mask(
        mask=al.Mask2D.circular(shape_native=(30, 30), pixel_scales=0.2, radius=2.2)
    )
    dataset = dataset.apply_sparse_operator(batch_size=16)

    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_basis.Basis(
            profile_list=[
                al.lp_linear.Gaussian(centre=(0.0, 0.0), sigma=sigma) for sigma in (0.2, 0.5, 1.0)
            ]
        ),
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=(8, 8)),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )

    fit = al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=[lens, source]),
        settings=al.Settings(use_border_relocator=True),
        xp=jnp,
    )
    inversion = fit.inversion

    sparse_ctx = sparse_steps.sparse_context_from(inversion=inversion, dataset=dataset)
    ctx = mfs.matrix_free_context_from(inversion=inversion, dataset=dataset, sparse_ctx=sparse_ctx)
    return fit, inversion, ctx


def test_matvec_reproduces_curvature_reg_matrix(tiny_inversion):
    _, inversion, ctx = tiny_inversion

    reference = np.asarray(inversion.curvature_reg_matrix, dtype=float)
    assert reference.shape == (ctx.total_params, ctx.total_params)

    identity = jnp.eye(ctx.total_params, dtype=jnp.float64)
    got = np.asarray(mfs.curvature_reg_matvec(ctx, identity), dtype=float)

    scale = max(float(np.max(np.abs(reference))), 1e-300)
    rel_diff = float(np.max(np.abs(got - reference))) / scale
    assert rel_diff <= 1e-8, (
        f"curvature_reg_matvec does not reproduce inversion.curvature_reg_matrix "
        f"(max rel diff {rel_diff:.3e} > 1e-8)"
    )

    # The 1-D signature returns 1-D and agrees with the batched one.
    column = np.asarray(mfs.curvature_reg_matvec(ctx, identity[:, 3]), dtype=float)
    assert column.shape == (ctx.total_params,)
    assert np.allclose(column, got[:, 3], rtol=0, atol=1e-12)

    # add_to_diag lands on the no-regularization (MGE) parameters, so F alone
    # must differ from F + λH only by the regularization matrix.
    curvature = np.asarray(mfs.curvature_matvec(ctx, identity), dtype=float)
    reg = np.asarray(inversion.regularization_matrix, dtype=float)
    assert np.allclose(curvature + reg, got, rtol=0, atol=1e-8 * scale)


def test_jacobi_diag_estimate_is_positive_and_exact_on_the_func_block(tiny_inversion):
    _, inversion, ctx = tiny_inversion

    diag = np.asarray(mfs.jacobi_diag_estimate(ctx), dtype=float)
    exact = np.diag(np.asarray(inversion.curvature_reg_matrix, dtype=float))

    assert diag.shape == (ctx.total_params,)
    assert np.all(diag > 0.0)

    # Exact where the basis is already PSF-operated (the func block); merely a
    # preconditioner on the mapper block, so only the order of magnitude is
    # asserted there.
    func = slice(ctx.sparse_ctx.func_start, ctx.sparse_ctx.func_end)
    assert np.allclose(diag[func], exact[func], rtol=1e-8)

    mapper = slice(ctx.sparse_ctx.mapper_start, ctx.sparse_ctx.mapper_end)
    live = exact[mapper] > 0.0
    ratio = diag[mapper][live] / exact[mapper][live]
    assert np.all(ratio > 1e-3) and np.all(ratio < 1e3)


def test_pcg_matches_cholesky_solve_on_the_inversion(tiny_inversion):
    _, inversion, ctx = tiny_inversion

    curvature_reg = jnp.asarray(inversion.curvature_reg_matrix, dtype=jnp.float64)
    data_vector = jnp.asarray(inversion.data_vector, dtype=jnp.float64)

    reference = reconstruction_steps.cholesky_solve(curvature_reg, data_vector)

    diag = mfs.jacobi_diag_estimate(ctx)
    x, iters, residual = mfs.pcg_solve(
        ctx, data_vector, tol=1e-14, maxiter=10 * ctx.total_params, diag=diag
    )

    assert 0 < int(iters) <= 10 * ctx.total_params
    assert float(residual) <= 1e-12

    scale = max(float(np.max(np.abs(np.asarray(reference)))), 1e-300)
    rel_diff = float(np.max(np.abs(np.asarray(x) - np.asarray(reference)))) / scale
    assert rel_diff <= 1e-8, f"PCG differs from the exact Cholesky solve ({rel_diff:.3e})"


def test_pdip_matrix_free_matches_the_cell_nnls(tiny_inversion):
    """Against ``reconstruction_steps.nnls_pdip``, not ``inversion.reconstruction``.

    With ``Settings.use_edge_zeroed_pixels`` on (the default, and on for this
    fixture — ``inversion.solve_ids_to_keep`` is a 39-of-67 subset here) the
    library does **not** solve the full system: it subsets ``F + λH`` and ``D``
    to the kept ids, runs the NNLS there and scatters zeros back. The exact
    comparator for a full-system solve is therefore the cell's own recipe —
    ``jacobi_scaled`` + ``nnls_pdip`` + rescale — which is what
    ``pdip_matrix_free`` mirrors.
    """
    _, inversion, ctx = tiny_inversion

    curvature_reg = jnp.asarray(inversion.curvature_reg_matrix, dtype=jnp.float64)
    data_vector = jnp.asarray(inversion.data_vector, dtype=jnp.float64)

    Q_pc, q_pc, scale = reconstruction_steps.jacobi_scaled(curvature_reg, data_vector)
    x_pc, converged_ref, iterations_ref = reconstruction_steps.nnls_pdip(Q_pc, q_pc)
    reference = np.asarray(x_pc * scale, dtype=float)
    assert int(converged_ref) == 1

    diag = mfs.jacobi_diag_estimate(ctx)
    x, pdip_iter, cg_iter, converged = mfs.pdip_matrix_free(
        ctx, data_vector, diag=diag, cg_tol=1e-12, cg_maxiter=10 * ctx.total_params
    )

    assert int(converged) == 1
    assert int(pdip_iter) > 1
    assert int(cg_iter) > int(pdip_iter)
    # Same iteration-count semantics: the inner solve is inexact and the
    # scaling is built from the estimated diagonal, so the path can differ by
    # a step but not by a regime.
    assert abs(int(pdip_iter) - int(iterations_ref)) <= 2

    reference_scale = max(float(np.max(np.abs(reference))), 1e-300)
    rel_diff = float(np.max(np.abs(np.asarray(x) - reference))) / reference_scale
    assert rel_diff <= 1e-6, (
        f"matrix-free PDIP differs from reconstruction_steps.nnls_pdip ({rel_diff:.3e})"
    )


def test_slq_logdet_matches_the_reduced_cholesky_log_dets(tiny_inversion):
    _, inversion, ctx = tiny_inversion

    n_reduced = ctx.mapper_params
    n_probes, n_lanczos = 32, n_reduced

    curvature_reg_reduced = jnp.asarray(inversion.curvature_reg_matrix_reduced, jnp.float64)
    reg_reduced = jnp.asarray(inversion.regularization_matrix_reduced, jnp.float64)

    exact_curvature_reg = float(reconstruction_steps.log_det_cholesky(curvature_reg_reduced))
    exact_reg = float(reconstruction_steps.log_det_cholesky(reg_reduced))

    key = jax.random.PRNGKey(0)

    for matvec, exact, label in (
        (mfs.reduced_matvec_from(ctx), exact_curvature_reg, "F+λH"),
        (
            mfs.reduced_matvec_from(ctx, mfs.bcoo_matvec(ctx.reg_bcoo)),
            exact_reg,
            "λH",
        ),
    ):
        estimate, per_probe, n_clamped = mfs.slq_logdet(
            matvec, n_reduced, n_probes=n_probes, n_lanczos=n_lanczos, key=key, reorth=True
        )
        assert int(n_clamped) == 0
        spread = float(np.std(np.asarray(per_probe), ddof=1)) / np.sqrt(n_probes)
        error = abs(float(estimate) - exact)
        # Even with n_lanczos == n the per-probe values still carry Hutchinson
        # noise, so the 3-sigma bound is the only assertable bar.
        assert error < 3.0 * spread + 1e-8, (
            f"SLQ log-det ({label}) error {error:.3e} exceeds 3 sigma ({3.0 * spread:.3e})"
        )


def test_matrix_free_evidence_terms_match_the_exact_terms(tiny_inversion):
    fit, inversion, ctx = tiny_inversion

    data = jnp.asarray(fit.dataset.data.array, dtype=jnp.float64)
    noise_map = jnp.asarray(fit.dataset.noise_map.array, dtype=jnp.float64)
    reconstruction = jnp.asarray(inversion.reconstruction, dtype=jnp.float64)
    model_data = jnp.asarray(fit.model_data.array, dtype=jnp.float64)

    exact = reconstruction_steps.log_evidence_terms(
        data=data,
        noise_map=noise_map,
        model_data=model_data,
        reconstruction=reconstruction,
        reduced_indices=jnp.asarray(np.asarray(inversion.mapper_indices)),
        reg_reduced=jnp.asarray(inversion.regularization_matrix_reduced, jnp.float64),
        curv_reg_reduced=jnp.asarray(inversion.curvature_reg_matrix_reduced, jnp.float64),
    )

    got = mfs.matrix_free_evidence_terms(
        ctx,
        jnp.asarray(inversion.data_vector, dtype=jnp.float64),
        reconstruction,
        log_det_curvature_reg=exact["log_det_curvature_reg"],
        log_det_regularization=exact["log_det_regularization"],
        data=data,
        noise_map=noise_map,
        model_data_fn=lambda _: model_data,
    )

    assert set(got) == set(exact)
    for key, value in exact.items():
        assert got[key] == pytest.approx(value, rel=1e-8, abs=1e-8), key
