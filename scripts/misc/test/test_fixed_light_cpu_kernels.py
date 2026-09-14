"""Unit tests for the CPU kernel rows and the phase-2 precision switches.

Phase 2 of the ``fixed-lens-light-profiling`` epic (autolens_profiling#253).
**NumPy only** — these rows exist to measure what the CPU costs *without* JAX,
so the tests that pin them import no JAX path beyond the reference solver they
compare against.

Three tiers, mirroring ``test_active_set_steps.py`` / ``test_fixed_light_library.py``:

1. **Dense QP, no imaging.** The three kernels must reach the *same* positive
   solution: the library's own numpy NNLS, the numpy certified active set, and
   the JAX PDIP reference the A100 rows were measured with. A CPU row that is
   fast because it solved a different problem is the failure this tier catches.

2. **A tiny ``FitImaging``.** The 30x30 / 8x8 / 3-linear-Gaussian fixture, run
   through ``cpu_kernel_rows`` end to end, pinning that every row is scored,
   that the thread block records **both** knob families, and that the equality
   gates are computed rather than asserted in prose.

3. **Static checks on the two cells.** ``--pins none`` must *withdraw verdicts,
   not skip comparisons*: the class of breakage where a leg quietly stops
   measuring something and still writes a JSON that looks complete.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_cpu_kernels.py
"""

from __future__ import annotations

import re
import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import active_set_steps as ass  # noqa: E402
from likelihood_breakdown import fixed_light_cpu_kernels as flck  # noqa: E402

KERNEL_CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_cpu_kernels.py"
LIBRARY_CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_library.py"


# ---------------------------------------------------------------------------
# 1. Dense QP — the three kernels are the same answer
# ---------------------------------------------------------------------------


def _constrained_qp(n: int, seed: int, negative_fraction: float = 0.35):
    """A Jacobi-scaled SPD QP whose NNLS solution has a non-trivial active set."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n + 5, n))
    Q = A.T @ A + 0.5 * np.eye(n)
    d = np.sqrt(np.diag(Q))
    Q = Q / d[:, None] / d[None, :]

    target = rng.standard_normal(n)
    target[rng.random(n) < negative_fraction] -= 1.5
    q = Q @ target
    return Q, q


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_library_numpy_nnls_equals_the_certified_active_set(seed):
    """The CPU production candidate and the CPU production kernel agree to 1e-8.

    The whole phase-2 CPU table rests on this: the certified active set is a
    *cheaper route to the same point*, not a different answer. If it were a
    different answer its milliseconds would be meaningless.
    """
    from autoarray.inversion.inversion import inversion_util

    Q, q = _constrained_qp(120, seed=seed)

    x_unc = ass.unconstrained_solve(Q, q)
    assert int(np.sum(x_unc < 0)) > 5, "QP is not actually constrained"

    x_library = np.asarray(
        inversion_util.reconstruction_positive_only_from(
            data_vector=q, curvature_reg_matrix=Q, settings=None, xp=np
        ),
        dtype=float,
    )

    result = ass.active_set_certified(Q, q, x_unc, policy="free_all", max_passes=60)
    assert result.certified, f"seed {seed} did not certify in {result.passes_run} passes"

    scale = max(float(np.max(np.abs(x_library))), 1e-300)
    assert float(np.max(np.abs(result.x - x_library))) / scale <= 1e-8

    # Both are genuinely feasible — a positive solution, not a signed one.
    assert float(np.min(x_library)) >= -1e-9 * scale
    assert float(np.min(result.x)) >= -1e-9 * scale


@pytest.mark.parametrize("seed", [0, 1])
def test_certified_active_set_equals_the_jax_pdip_reference(seed):
    """...and both equal ``solve_nnls``, the solver the A100 rows were measured with."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from likelihood_breakdown import reconstruction_steps

    Q, q = _constrained_qp(100, seed=seed)
    x_unc = ass.unconstrained_solve(Q, q)
    result = ass.active_set_certified(Q, q, x_unc, policy="free_all", max_passes=60)
    assert result.certified

    x_pdip, _, _ = jax.jit(reconstruction_steps.nnls_pdip)(
        jnp.asarray(Q, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)
    )
    x_pdip = np.asarray(x_pdip, dtype=float)

    scale = max(float(np.max(np.abs(x_pdip))), 1e-300)
    assert float(np.max(np.abs(result.x - x_pdip))) / scale <= 1e-8


def test_scipy_unconstrained_is_the_plain_solve():
    """The floor row really is the unconstrained solve, not a clipped one."""
    Q, q = _constrained_qp(80, seed=3)
    x = ass.unconstrained_solve(Q, q)
    assert np.allclose(x, np.linalg.solve(Q, q), rtol=0, atol=1e-10)
    assert int(np.sum(x < 0)) > 0, "an unconstrained row with no negatives proves nothing"


# ---------------------------------------------------------------------------
# The timing helper and the thread block
# ---------------------------------------------------------------------------


def test_time_median_reports_the_sample_it_took():
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1

    out = flck.time_median(_fn, n_repeats=5, n_warmup=2)
    assert calls["n"] == 7, "warm-up calls must run and be discarded, not skipped"
    assert out["n_repeats"] == 5
    assert out["n_warmup"] == 2
    assert out["ms_min"] <= out["ms"] <= out["ms_max"]
    assert "median" in out["timer"]


def test_time_median_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        flck.time_median(lambda: None, n_repeats=0)


def test_thread_block_records_both_disjoint_knob_families(monkeypatch):
    """A CPU millisecond needs both knobs — NPROC (JAX) and the BLAS pins.

    They are disjoint: ``NPROC`` sizes XLA's CPU intra-op pool and does nothing
    to OpenBLAS; the BLAS pins do nothing to JAX. Recording one and inferring
    the other is how a CPU row gets misread by a factor of the core count.
    """
    monkeypatch.setenv("NPROC", "8")
    for name in flck.BLAS_THREAD_VARS:
        monkeypatch.setenv(name, "1")

    block = flck.thread_block()

    assert set(block["blas"]) == set(flck.BLAS_THREAD_VARS)
    assert set(block["jax_pool"]) == set(flck.JAX_THREAD_VARS)
    assert all(v == "1" for v in block["blas"].values())
    assert block["n_threads_blas"] == 1
    assert block["n_threads_jax_pool"] == 8
    assert not set(flck.BLAS_THREAD_VARS) & set(flck.JAX_THREAD_VARS), (
        "the families must be disjoint"
    )


def test_thread_block_survives_unset_knobs(monkeypatch):
    """An unset knob is ``None``, not a guess and not a crash."""
    for name in flck.BLAS_THREAD_VARS + flck.JAX_THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    block = flck.thread_block()
    assert block["n_threads_blas"] is None
    assert block["n_threads_jax_pool"] is None


# ---------------------------------------------------------------------------
# 2. A tiny FitImaging — the rows end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_s3_system():
    """The 67-parameter fixture of ``test_active_set_steps.py``, as an S3 system."""
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
    return ass.fixed_light_system_from(fit, dataset, name="S3")


def test_solver_view_is_the_subset_the_library_hands_its_solver(tiny_s3_system):
    """Under edge zeroing the solver never sees the border pixels at all."""
    view = flck.solver_view_of(tiny_s3_system)
    n_edge = int(tiny_s3_system.edge_zero_mask.sum())

    assert view["n_full"] == int(tiny_s3_system.n_params)
    assert view["n_seen_by_solver"] == view["n_full"] - n_edge
    assert view["curvature_reg_matrix"].shape == (
        view["n_seen_by_solver"],
        view["n_seen_by_solver"],
    )

    # The scatter-back is the library's: exact zeros everywhere it did not solve.
    x_sub = np.ones(view["n_seen_by_solver"], dtype=float)
    full = view["scatter_back"](x_sub)
    assert full.shape == (view["n_full"],)
    assert int(np.sum(full == 0.0)) == n_edge


def test_cpu_kernel_rows_scores_every_row(tiny_s3_system):
    """No bare milliseconds: every row carries its evidence and its negatives."""
    out = flck.cpu_kernel_rows(tiny_s3_system, pass_budget=7, n_repeats=2)

    rows = [
        out["rows"]["library_numpy_nnls"],
        out["rows"]["scipy_unconstrained"]["subset"],
        out["rows"]["scipy_unconstrained"]["full_system"],
        out["rows"]["numpy_certified_active_set"],
    ]
    for row in rows:
        for key in ("ms", "log_evidence", "d_log_evidence_vs_library", "n_negative_entries", "n"):
            assert key in row, f"{row['label']} is missing {key}"
        assert row["ms"] > 0.0


def test_cpu_kernel_rows_equalities_are_computed_and_pass(tiny_s3_system):
    """The two positivity rows reach the library's own answer — measured, not claimed."""
    out = flck.cpu_kernel_rows(tiny_s3_system, pass_budget=7, n_repeats=2)

    assert len(out["equalities"]) == 2
    for entry in out["equalities"]:
        assert "rel_diff" in entry and "rtol" in entry
        assert entry["status"] == "PASS", f"{entry['check']} drifted by {entry['rel_diff']:.3e}"

    assert out["rows"]["numpy_certified_active_set"]["certified"] is True
    assert out["rows"]["library_numpy_nnls"]["n_negative_entries"] == 0
    assert out["rows"]["numpy_certified_active_set"]["n_negative_entries"] == 0


def test_the_unconstrained_row_is_reported_with_its_cost(tiny_s3_system):
    """Dropping positivity buys milliseconds and costs nats — both must be in the row."""
    out = flck.cpu_kernel_rows(tiny_s3_system, pass_budget=7, n_repeats=2)
    unconstrained = out["rows"]["scipy_unconstrained"]["full_system"]
    nnls = out["rows"]["library_numpy_nnls"]

    assert unconstrained["ms"] < nnls["ms"], "the unconstrained floor should be the cheapest row"
    assert unconstrained["n_negative_entries"] > 0, (
        "an unconstrained row with no negative pixels is not exercising the difference"
    )
    # A POSITIVE delta is not an improvement: it is a different, infeasible minimiser.
    assert unconstrained["d_log_evidence_vs_library"] > 0.0


def test_cpu_kernel_rows_records_the_threads_it_ran_at(tiny_s3_system):
    out = flck.cpu_kernel_rows(tiny_s3_system, pass_budget=7, n_repeats=1)
    assert set(out["threads"]["blas"]) == set(flck.BLAS_THREAD_VARS)
    assert set(out["threads"]["jax_pool"]) == set(flck.JAX_THREAD_VARS)


def test_cpu_kernel_rows_is_json_serialisable(tiny_s3_system):
    """The cell writes this dict straight to disk — no stray callables or arrays."""
    import json

    out = flck.cpu_kernel_rows(tiny_s3_system, pass_budget=7, n_repeats=1)
    json.dumps(out)


# ---------------------------------------------------------------------------
# 3. Static checks — --pins none withdraws verdicts, it does not skip work
# ---------------------------------------------------------------------------


def test__the_kernel_cell_exists_and_declares_its_flags():
    assert KERNEL_CELL.is_file(), f"{KERNEL_CELL} is missing"
    text = KERNEL_CELL.read_text()
    for flag in ("--mesh", "--pass-budget", "--source-pixels", "--n-repeats"):
        assert flag in text, f"the kernel cell does not declare {flag}"


def test__the_kernel_cell_carries_phase_0s_certifying_budgets():
    text = KERNEL_CELL.read_text()
    match = re.search(r"CERTIFYING_BUDGET = \{([^}]*)\}", text)
    assert match, "the kernel cell does not declare CERTIFYING_BUDGET"
    for mesh, budget in (("rectangular", 7), ("delaunay", 2), ("delaunay_nn", 2)):
        assert f'"{mesh}": {budget}' in match.group(1)


def test__the_library_cell_declares_the_precision_switch():
    text = LIBRARY_CELL.read_text()
    assert '"--pins", choices=("fp64", "none"), default="fp64"' in text, (
        "the default must stay fp64 — the A100 legs must not change behaviour"
    )
    assert "--use-mixed-precision" in text


def test__pins_none_records_rather_than_skips():
    """The failure this guards: a leg that stops comparing and still looks complete.

    ``--pins none`` exists because a pin calibrated in fp64 is not a pin in
    fp32. It must withdraw the *verdict* and keep the *measurement*: every
    comparison still computed, written with status ``RECORDED``.
    """
    text = LIBRARY_CELL.read_text()
    assert "PINS_ASSERT = PINS_MODE ==" in text
    assert '"RECORDED"' in text, "the withdrawn verdicts must still be written as data"
    assert "pins_recorded" in text, "the fp64 pins must still be computed and recorded"
    # The mapper-block identity raises only when the pins are being asserted.
    assert "if PINS_ASSERT and _rel > _MAPPER_LOGDET_RTOL:" in text


def test__the_library_cell_re_derives_tau_rel_per_precision():
    """A tolerance calibrated in fp64 is not a tolerance in fp32."""
    text = LIBRARY_CELL.read_text()
    assert "TAU_REL_BASIS" in text
    assert "_EPS_F32" in text
    assert "tau_rel=TAU_REL" in text, "the injected solver must run at the re-derived tolerance"


def test__both_cells_record_the_machine():
    for cell in (KERNEL_CELL, LIBRARY_CELL):
        text = cell.read_text()
        assert "machine_info_dict" in text, f"{cell.name} does not record the machine block"
