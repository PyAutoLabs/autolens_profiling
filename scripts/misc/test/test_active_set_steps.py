"""Unit tests for the certified active-set kernels (``active_set_steps.py``).

CPU only, fp64. Two tiers, mirroring ``test_matrix_free_steps.py``:

1. **Dense references, no imaging.** The schemes take a plain ``(Q, q)`` QP, so
   the certified numpy solution is checked against
   ``autoarray.util.jax_nnls.solve_nnls`` on random SPD systems with a genuinely
   active constraint set, and the masked-JAX twin is checked pass by pass
   against the numpy restricted solve. These pin the *algorithms*.

2. **A tiny ``FitImaging`` with linear lens light.** A 30x30 simulated dataset,
   an 8x8 ``RectangularUniform`` mesh and a 3-Gaussian **linear** basis give a
   67-parameter S0 inversion; ``fixed_light_system_from`` converts that light to
   regular profiles, subtracts it and rebuilds the source-only system (S3). This
   pins the *claim the whole task rests on*: fixing the lens light at its solved
   intensities changes neither the evidence nor the mapper-block log-dets, and
   the edge-zero set the schemes hold at zero is the library's own.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_active_set_steps.py
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

from likelihood_breakdown import active_set_steps as ass  # noqa: E402
from likelihood_breakdown import reconstruction_steps  # noqa: E402

# ---------------------------------------------------------------------------
# Dense QP helpers
# ---------------------------------------------------------------------------


def _constrained_qp(n: int, seed: int, negative_fraction: float = 0.35):
    """A Jacobi-scaled SPD QP whose NNLS solution has a non-trivial active set.

    ``q`` is built from a deliberately *signed* target so the unconstrained
    solve puts a sizeable fraction of the entries below zero — a QP with no
    active constraints would pass every test here vacuously.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n + 5, n))
    Q = A.T @ A + 0.5 * np.eye(n)
    d = np.sqrt(np.diag(Q))
    Q = Q / d[:, None] / d[None, :]

    target = rng.standard_normal(n)
    target[rng.random(n) < negative_fraction] -= 1.5
    q = Q @ target
    return Q, q


def _restricted_solve(Q, q, Z):
    """``x_F = Q[F,F]⁻¹ q_F``, ``x_Z = 0`` — the numpy restricted solve."""
    x = np.zeros(q.shape[0], dtype=float)
    free = ~np.asarray(Z, dtype=bool)
    idx = np.where(free)[0]
    if idx.size:
        x[idx] = np.linalg.solve(Q[np.ix_(idx, idx)], q[idx])
    return x


# ---------------------------------------------------------------------------
# 1. The certified numpy scheme against the library's PDIP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_certified_solution_matches_solve_nnls(seed):
    Q, q = _constrained_qp(120, seed=seed)

    x_unc = ass.unconstrained_solve(Q, q)
    assert int(np.sum(x_unc < 0)) > 5, "QP is not actually constrained"

    result = ass.active_set_certified(Q, q, x_unc, policy="free_all", max_passes=60)
    assert result.certified, f"seed {seed} did not certify in {result.passes_run} passes"

    x_pdip, _, _ = jax.jit(reconstruction_steps.nnls_pdip)(
        jnp.asarray(Q, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)
    )
    x_pdip = np.asarray(x_pdip, dtype=float)

    scale = max(float(np.max(np.abs(x_pdip))), 1e-300)
    assert float(np.max(np.abs(result.x - x_pdip))) / scale <= 1e-8

    # The certificate is a statement about the KKT conditions, not a tolerance:
    # the final iterate is feasible and its gradient is non-negative where fixed.
    g = Q @ result.x - q
    assert float(np.min(result.x)) >= -1e-9 * scale
    assert float(np.min(g[result.fixed_set])) >= -1e-9 * float(np.max(np.abs(q)))


def test_free_one_reaches_the_same_solution():
    """The one-at-a-time release policy is slower, not different."""
    Q, q = _constrained_qp(80, seed=7)
    x_unc = ass.unconstrained_solve(Q, q)

    all_ = ass.active_set_certified(Q, q, x_unc, policy="free_all", max_passes=200)
    one = ass.active_set_certified(Q, q, x_unc, policy="free_one", max_passes=200)

    assert all_.certified and one.certified
    scale = max(float(np.max(np.abs(all_.x))), 1e-300)
    assert float(np.max(np.abs(all_.x - one.x))) / scale <= 1e-9
    assert one.n_factorisations >= all_.n_factorisations


def test_permanent_fixed_set_is_never_released():
    """``fixed0`` indices stay at zero however negative their gradient gets."""
    Q, q = _constrained_qp(60, seed=3)
    fixed0 = np.zeros(60, dtype=bool)
    fixed0[np.argsort(-q)[:5]] = True  # the five most *wanted* entries

    x_unc = ass.unconstrained_solve(Q, q, fixed0=fixed0)
    assert np.all(x_unc[fixed0] == 0.0)

    result = ass.active_set_certified(Q, q, x_unc, fixed0=fixed0, max_passes=60)
    assert result.certified
    assert np.all(result.x[fixed0] == 0.0)
    assert np.all(result.fixed_set[fixed0])

    # And the certificate is genuinely for the restricted problem: the full
    # problem's solution differs.
    free_result = ass.active_set_certified(Q, q, ass.unconstrained_solve(Q, q), max_passes=60)
    assert float(np.max(np.abs(free_result.x - result.x))) > 1e-6


def test_log_evidence_hook_is_called_once_per_iterate():
    """The hook records, it does not steer: one value per iterate, pass 0 included."""
    Q, q = _constrained_qp(60, seed=9)
    seen: list[np.ndarray] = []

    def hook(x):
        seen.append(np.asarray(x).copy())
        return float(np.sum(x))

    result = ass.active_set_certified(
        Q, q, ass.unconstrained_solve(Q, q), max_passes=60, log_evidence_fn=hook
    )
    assert result.certified
    assert len(result.log_evidence_per_pass) == result.passes_run + 1
    assert len(seen) == len(result.iterates)
    for recorded, iterate in zip(seen, result.iterates):
        assert np.array_equal(recorded, iterate)


def test_warm_start_skips_pass_zero():
    Q, q = _constrained_qp(80, seed=5)
    cold = ass.active_set_certified(Q, q, ass.unconstrained_solve(Q, q), max_passes=60)
    assert cold.certified

    warm = ass.active_set_certified(Q, q, None, initial_fixed=cold.fixed_set, max_passes=60)
    assert warm.warm_start
    assert warm.certified
    assert warm.passes_to_certification == 1
    assert warm.factorisations_after_pass[0] == 0
    assert np.allclose(warm.x, cold.x, rtol=1e-9, atol=1e-11)


def test_a_wrong_warm_start_is_released_not_certified():
    """``initial_fixed`` is a guess, and a guess must be releasable.

    Seeding the *permanent* ``fixed0`` with a neighbouring point's active set
    would certify the restricted problem instead of the QP, and hand back a
    wrong answer wearing a certificate — measured at 0.1 relative error and
    −8 nats on the probe's +-1 % mass draws before this distinction existed.
    """
    Q, q = _constrained_qp(80, seed=13)
    truth = ass.active_set_certified(Q, q, ass.unconstrained_solve(Q, q), max_passes=200)
    assert truth.certified

    wrong = truth.fixed_set.copy()
    free_indices = np.where(~truth.fixed_set)[0]
    wrong[free_indices[:10]] = True  # ten indices that belong in the solution

    released = ass.active_set_certified(Q, q, None, initial_fixed=wrong, max_passes=200)
    assert released.certified
    assert np.allclose(released.x, truth.x, rtol=1e-8, atol=1e-10)

    pinned = ass.active_set_certified(Q, q, None, fixed0=wrong, max_passes=200)
    assert pinned.certified  # certified — for a different problem
    assert float(np.max(np.abs(pinned.x - truth.x))) > 1e-6


# ---------------------------------------------------------------------------
# 2. The masked-JAX twin against the numpy reference
# ---------------------------------------------------------------------------


def test_masked_jax_pass_matches_numpy_restricted_solve():
    """One masked full-size Cholesky == the numpy restricted solve on the same Z."""
    Q, q = _constrained_qp(90, seed=11)
    fixed0 = np.zeros(90, dtype=bool)
    fixed0[:7] = True

    out = jax.jit(ass.active_set_masked_jax, static_argnums=(3,))(
        jnp.asarray(Q), jnp.asarray(q), jnp.asarray(fixed0), 1
    )

    x0 = np.asarray(out["x_unconstrained"], dtype=float)
    assert np.allclose(x0, ass.unconstrained_solve(Q, q, fixed0=fixed0), rtol=1e-10, atol=1e-12)

    Z0 = fixed0 | (x0 < 0.0)
    x1 = np.asarray(out["xs"][1], dtype=float)
    assert np.allclose(x1, _restricted_solve(Q, q, Z0), rtol=1e-9, atol=1e-12)
    assert np.all(x1[Z0] == 0.0)
    assert int(out["n_fixed"][0]) == int(Z0.sum())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_masked_jax_certifies_at_the_numpy_pass_count(seed):
    Q, q = _constrained_qp(120, seed=seed)
    x_unc = ass.unconstrained_solve(Q, q)
    reference = ass.active_set_certified(Q, q, x_unc, policy="free_all", max_passes=60)
    assert reference.certified

    budget = reference.passes_to_certification + 2
    out = jax.jit(ass.active_set_masked_jax, static_argnums=(3,))(
        jnp.asarray(Q), jnp.asarray(q), jnp.zeros(120, dtype=bool), budget
    )

    certified = np.asarray(out["certified"])
    assert certified.any()
    # scan pass i is the reference's pass i + 1.
    assert int(np.argmax(certified)) + 1 == reference.passes_to_certification

    # Same iterate at every pass, not only at the end.
    for k in range(1, reference.passes_to_certification + 1):
        assert np.allclose(
            np.asarray(out["xs"][k], dtype=float), reference.iterates[k], rtol=1e-9, atol=1e-11
        )

    # Once certified the fixed set is frozen, so the leftover budget is idempotent.
    assert np.allclose(
        np.asarray(out["x"], dtype=float), reference.x, rtol=1e-9, atol=1e-11
    )


def test_masked_jax_violation_counts_match_the_reference():
    Q, q = _constrained_qp(100, seed=4)
    reference = ass.active_set_certified(
        Q, q, ass.unconstrained_solve(Q, q), policy="free_all", max_passes=60
    )
    out = ass.active_set_masked_jax(
        jnp.asarray(Q), jnp.asarray(q), jnp.zeros(100, dtype=bool), reference.passes_run
    )
    for k, entry in enumerate(reference.history):
        assert int(out["n_fixed"][k]) == entry["n_fixed"]
        assert int(out["n_primal_violations"][k]) == entry["n_primal_violations"]
        assert int(out["n_dual_violations"][k]) == entry["n_dual_violations"]


# ---------------------------------------------------------------------------
# 3. The fixed-lens-light system, against a tiny FitImaging
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_s0_fit():
    """A 67-parameter S0 fit: 3 **linear** Gaussians of lens light + an 8x8 mesh.

    The same geometry ``test_matrix_free_steps.py`` uses, minus the sparse
    operator — ``fixed_light_system_from`` builds a dense inversion by design
    (the w-tilde weight map is baked from the unsubtracted image).
    """
    autolens = pytest.importorskip("autolens")
    al = autolens

    grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.2)
    psf = al.Convolver.from_gaussian(shape_native=(5, 5), sigma=0.3, pixel_scales=0.2)

    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, background_sky_level=0.1, noise_seed=1
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
        xp=np,
    )
    return fit, dataset


@pytest.fixture(scope="module")
def tiny_systems(tiny_s0_fit):
    fit, dataset = tiny_s0_fit
    s0 = ass.linear_system_from(fit, dataset, name="S0")
    s3 = ass.fixed_light_system_from(fit, dataset, name="S3")
    return s0, s3


def test_system_model_data_reproduces_the_library(tiny_systems):
    """``blurred_image + Λ x`` is ``FitImaging.model_data``, and scores its evidence."""
    s0, _ = tiny_systems
    x_lib = np.asarray(s0.inversion.reconstruction, dtype=float)

    model_data = s0.model_data_from(x_lib)
    assert np.allclose(
        model_data, np.asarray(s0.fit.model_data.array, dtype=float), rtol=0, atol=1e-12
    )
    assert s0.log_evidence(x_lib) == pytest.approx(float(s0.fit.log_evidence), rel=1e-12)


def test_source_only_system_has_no_linear_func_list(tiny_systems):
    s0, s3 = tiny_systems
    assert s0.n_funcs == 3
    assert s3.n_funcs == 0
    assert s3.n_params == s3.n_mapper == s0.n_mapper
    assert s3.subtracted_light_flux is not None and s3.subtracted_light_flux != 0.0


def test_fixing_the_light_preserves_the_evidence(tiny_systems):
    """S3's evidence is S0's: fixing the light at its solved intensities is free.

    The joint optimum ``(a*, s*)`` is by definition the ``s``-optimum given
    ``a*``, so re-solving the source block with the light frozen must return the
    same reconstruction, the same model image and the same evidence — provided
    the library's linear-to-regular conversion really does reproduce
    ``Λ_mge a*``. That conversion is what this test pins.
    """
    s0, s3 = tiny_systems
    ev0 = s0.log_evidence_terms(np.asarray(s0.inversion.reconstruction, dtype=float))
    ev3 = s3.log_evidence_terms(np.asarray(s3.inversion.reconstruction, dtype=float))

    assert ev3["log_evidence"] == pytest.approx(ev0["log_evidence"], rel=1e-8)
    assert ev3["chi_squared"] == pytest.approx(ev0["chi_squared"], rel=1e-8)
    assert ev3["regularization_term"] == pytest.approx(ev0["regularization_term"], rel=1e-8)


def test_mapper_block_log_dets_are_unchanged(tiny_systems):
    """Both log-det terms are the mapper block's, and the mapper block is the same matrix."""
    s0, s3 = tiny_systems
    for name in ("curv_reg_reduced", "reg_reduced"):
        a = getattr(s0, name)
        b = getattr(s3, name)
        assert a.shape == b.shape
        log_det_a = float(reconstruction_steps.log_det_cholesky(jnp.asarray(a)))
        log_det_b = float(reconstruction_steps.log_det_cholesky(jnp.asarray(b)))
        assert log_det_b == pytest.approx(log_det_a, rel=1e-9)


def test_edge_zero_mask_is_the_librarys_zeroed_reconstruction(tiny_systems):
    """The mask handed to the schemes is exactly what the library holds at zero."""
    for system in tiny_systems:
        ids = system.inversion.solve_ids_to_keep
        assert ids is not None, "the rectangular fixture must exercise edge zeroing"

        expected = np.ones(system.n_params, dtype=bool)
        expected[np.asarray(ids, dtype=int)] = False
        assert np.array_equal(system.edge_zero_mask, expected)
        assert int(system.edge_zero_mask.sum()) > 0

        # Every edge-zeroed index is exactly zero in the library's reconstruction.
        # The converse does not hold and must not be asserted: on the numpy path
        # the positive-only solver is fnnls, whose *own* active set also comes
        # out exactly zero, so `reconstruction == 0` is a strict superset.
        reconstruction = np.asarray(system.inversion.reconstruction, dtype=float)
        assert np.all(reconstruction[system.edge_zero_mask] == 0.0)
        assert np.all((reconstruction == 0.0)[system.edge_zero_mask])


def test_certified_scheme_reproduces_the_library_reconstruction(tiny_systems):
    """With the edge-zero set fixed, the certified solve *is* the library's answer."""
    _, s3 = tiny_systems

    x_unc = ass.unconstrained_solve(s3.Q, s3.q, fixed0=s3.edge_zero_mask)
    result = ass.active_set_certified(
        s3.Q, s3.q, x_unc, fixed0=s3.edge_zero_mask, max_passes=60
    )
    assert result.certified

    x = result.x * s3.d_scale
    reference = np.asarray(s3.inversion.reconstruction, dtype=float)
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    assert float(np.max(np.abs(x - reference))) / scale <= 1e-6
    assert np.all(x[s3.edge_zero_mask] == 0.0)

    # Scored against the library's own reconstruction — NOT against the PDIP
    # solution, which solves the *unzeroed* system and therefore sits at a
    # different evidence (see the next test).
    delta = ass.active_set_evidence_error(
        s3, x, reference_log_evidence=s3.log_evidence(reference)
    )
    assert abs(delta) <= 1e-6


def test_certified_scheme_without_the_edge_zero_seed_matches_pdip(tiny_systems):
    """Dropped ``fixed0``: the certificate is then for the pure NNLS problem.

    The two are different problems, and the gap is not small — the edge-zeroed
    solve gives up every border pixel — so the probe reports both.
    """
    _, s3 = tiny_systems

    result = ass.active_set_certified(
        s3.Q, s3.q, ass.unconstrained_solve(s3.Q, s3.q), fixed0=None, max_passes=60
    )
    assert result.certified

    x = result.x * s3.d_scale
    x_pdip = s3.pdip()[0]
    scale = max(float(np.max(np.abs(x_pdip))), 1e-300)
    assert float(np.max(np.abs(x - x_pdip))) / scale <= 1e-6
    assert abs(ass.active_set_evidence_error(s3, x)) <= 1e-6

    edge_zeroed = np.asarray(s3.inversion.reconstruction, dtype=float)
    assert s3.log_evidence(x_pdip) - s3.log_evidence(edge_zeroed) > 1.0


def test_masked_jax_matches_the_reference_on_the_real_system(tiny_systems):
    _, s3 = tiny_systems
    x_unc = ass.unconstrained_solve(s3.Q, s3.q, fixed0=s3.edge_zero_mask)
    reference = ass.active_set_certified(
        s3.Q, s3.q, x_unc, fixed0=s3.edge_zero_mask, max_passes=60
    )
    assert reference.certified

    out = jax.jit(ass.active_set_masked_jax, static_argnums=(3,))(
        jnp.asarray(s3.Q),
        jnp.asarray(s3.q),
        jnp.asarray(s3.edge_zero_mask),
        reference.passes_to_certification + 1,
    )
    certified = np.asarray(out["certified"])
    assert int(np.argmax(certified)) + 1 == reference.passes_to_certification
    assert np.allclose(np.asarray(out["x"], dtype=float), reference.x, rtol=1e-8, atol=1e-10)


def test_evidence_error_of_the_pdip_solution_is_zero(tiny_systems):
    _, s3 = tiny_systems
    x_pdip, converged, iters = s3.pdip()
    assert converged and iters > 0
    assert ass.active_set_evidence_error(s3, x_pdip) == pytest.approx(0.0, abs=1e-9)
