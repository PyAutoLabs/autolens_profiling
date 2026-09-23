"""Unit tests for the library-path solver injection and the ``fixed_light_library`` cell.

Three tiers, mirroring ``test_active_set_steps.py`` + ``test_fixed_light_cell.py``:

1. **Dense QP, no imaging.** The injected wrapper is a drop-in for the library's
   own ``reconstruction_positive_only_from``, so it is checked against that
   function on random SPD systems with a genuinely active constraint set: it
   must return the library's PDIP answer when the budget certifies, return it
   *exactly* when the budget is exhausted and the fallback fires, and return a
   demonstrably different (uncertified) iterate when the fallback is removed.

2. **A tiny ``FitImaging``.** The 30x30 / 8x8 / 3-linear-Gaussian fixture of
   ``test_active_set_steps.py``, used to pin the two facts about the *library*
   this cell's routes depend on: that the patch is actually reached through
   ``Inversion.reconstruction``, and that ``use_positive_only_solver=False``
   really does route to ``reconstruction_positive_negative_from`` — with edge
   zeroing silently disabled — and give the plain ``np.linalg.solve`` answer.

3. **Static checks on the cell and its A100 submits**, the class of breakage
   that has actually happened in this repo (a submit passing a flag the
   checked-out cell did not have; ``parse_known_args`` swallowed it and the run
   silently produced a different table).

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_library.py
"""

from __future__ import annotations

import inspect
import re
import subprocess
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


ROOT = _profiling_root()
_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import library_solver_injection as lsi  # noqa: E402

CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_library.py"
BATCH_GPU = ROOT / "hpc" / "batch_gpu"
LAUNCHER = BATCH_GPU / "submit_fixed_light_library.sh"

MESHES = ("rectangular", "delaunay", "delaunay_nn")
CONFIG_NAME = "hpc_a100_fp64_fixed_light_library"


# ---------------------------------------------------------------------------
# 1. The wrapper against the library's own positive-only solve
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
    return Q, Q @ target


def _library_positive_only():
    from autoarray.inversion.inversion import inversion_util

    return inversion_util.reconstruction_positive_only_from


def _solve(Q, q, *, n_passes, fallback):
    original = _library_positive_only()

    def fn(Q_, q_):
        return lsi.certified_reconstruction_from(
            q_, Q_, n_passes=n_passes, fallback=fallback, original=original, xp=jnp
        )

    return np.asarray(
        jax.jit(fn)(jnp.asarray(Q, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)),
        dtype=float,
    )


def _library_answer(Q, q):
    original = _library_positive_only()

    def fn(Q_, q_):
        return original(data_vector=q_, curvature_reg_matrix=Q_, settings=None, xp=jnp)

    return np.asarray(
        jax.jit(fn)(jnp.asarray(Q, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)),
        dtype=float,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_injected_solver_matches_the_library_when_it_certifies(seed):
    """A generous budget certifies, and the certified iterate IS the library's answer."""
    Q, q = _constrained_qp(120, seed=seed)
    reference = _library_answer(Q, q)
    scale = max(float(np.max(np.abs(reference))), 1e-300)

    for fallback in (True, False):
        got = _solve(Q, q, n_passes=40, fallback=fallback)
        assert float(np.max(np.abs(got - reference))) / scale <= 1e-8, (
            f"seed {seed}, fallback={fallback}: the certified solve did not reproduce "
            f"the library's positive-only solution"
        )
        assert float(np.min(got)) >= -1e-9 * scale, "the certified iterate is infeasible"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exhausted_budget_falls_back_to_the_library_exactly(seed):
    """Budget 1 never certifies on these QPs, so the fallback must return PDIP itself.

    ``lax.cond`` selects a branch rather than blending, so the fallback answer is
    the library's own array bit-for-bit — an exact equality, not a tolerance.
    """
    Q, q = _constrained_qp(120, seed=seed)
    reference = _library_answer(Q, q)

    uncertified = _solve(Q, q, n_passes=1, fallback=False)
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    assert float(np.max(np.abs(uncertified - reference))) / scale > 1e-3, (
        "budget 1 certified on this QP, so the fallback test below would be vacuous"
    )

    fell_back = _solve(Q, q, n_passes=1, fallback=True)
    assert np.array_equal(fell_back, reference)


def test_context_manager_restores_the_library_function():
    from autoarray.inversion.inversion import inversion_util

    before = inversion_util.reconstruction_positive_only_from
    with lsi.certified_solver_injected(3) as counts:
        assert inversion_util.reconstruction_positive_only_from is not before
        assert counts == {"jax": 0, "numpy": 0}
    assert inversion_util.reconstruction_positive_only_from is before


def test_context_manager_restores_on_an_exception():
    from autoarray.inversion.inversion import inversion_util

    before = inversion_util.reconstruction_positive_only_from
    with pytest.raises(RuntimeError), lsi.certified_solver_injected(3):
        raise RuntimeError("boom")
    assert inversion_util.reconstruction_positive_only_from is before


def test_numpy_path_is_delegated_untouched():
    """A NumPy-path call inside the patch is the library's, not the active set.

    The cell builds eager (``xp=np``) fits for its diagnostics; if the patch
    caught those too, a reference number would silently become an active-set
    number.
    """
    Q, q = _constrained_qp(60, seed=5)
    original = _library_positive_only()
    expected = original(data_vector=q, curvature_reg_matrix=Q, settings=None, xp=np)

    from autoarray.inversion.inversion import inversion_util

    with lsi.certified_solver_injected(1, fallback=False) as counts:
        got = inversion_util.reconstruction_positive_only_from(
            data_vector=q, curvature_reg_matrix=Q, settings=None, xp=np
        )
    assert counts == {"jax": 0, "numpy": 1}
    assert np.allclose(np.asarray(got, dtype=float), np.asarray(expected, dtype=float))


def _stub_new(calls):
    def stub(
        data_vector,
        curvature_reg_matrix,
        settings=None,
        xp=np,
        fingerprint=None,
        factor=None,
        solver="pdip",
        stats=None,
    ):
        calls.append({"solver": solver, "stats": stats})
        return data_vector

    return stub


def _stub_old(calls):
    def stub(
        data_vector, curvature_reg_matrix, settings=None, xp=np, fingerprint=None, factor=None
    ):
        calls.append({})
        return data_vector

    return stub


@pytest.mark.parametrize("new_signature", [True, False])
def test_injection_installs_against_the_old_and_new_library_signature(monkeypatch, new_signature):
    """PyAutoArray #566 added ``solver``/``stats``; the harness must run on either side of it."""
    from autoarray.inversion.inversion import inversion_util

    calls = []
    stub = (_stub_new if new_signature else _stub_old)(calls)
    monkeypatch.setattr(inversion_util, "reconstruction_positive_only_from", stub)

    q = np.ones(4)
    stats = {}
    with lsi.certified_solver_injected(3) as counts:
        inversion_util.reconstruction_positive_only_from(
            data_vector=q, curvature_reg_matrix=np.eye(4), xp=np, solver="certified", stats=stats
        )
    assert counts == {"jax": 0, "numpy": 1}
    if new_signature:
        assert calls == [{"solver": "certified", "stats": stats}]
    else:
        assert calls == [{}], "the old library must not be handed solver/stats"
    assert inversion_util.reconstruction_positive_only_from is stub


def test_fallback_pins_the_library_pdip_solver():
    """Route (d)'s fallback is the library's PDIP, whatever solver the library config selects."""
    original = _library_positive_only()
    assert "solver" in inspect.signature(original).parameters, (
        "this test needs the PyAutoArray #566 library on the path (source activate.sh)"
    )
    seen = []

    def spy(
        data_vector,
        curvature_reg_matrix,
        settings=None,
        xp=np,
        fingerprint=None,
        factor=None,
        solver="<absent>",
        stats="<absent>",
    ):
        seen.append((solver, stats))
        return original(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            settings=settings,
            xp=xp,
            fingerprint=fingerprint,
            factor=factor,
            solver=solver,
            stats=stats,
        )

    Q, q = _constrained_qp(60, seed=0)
    jax.jit(
        lambda Q_, q_: lsi.certified_reconstruction_from(
            q_, Q_, n_passes=1, fallback=True, original=spy, xp=jnp
        )
    )(jnp.asarray(Q), jnp.asarray(q))
    assert seen and all(s == ("pdip", None) for s in seen)


# ---------------------------------------------------------------------------
# 2. The two library facts the cell's routes stand on
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_fit():
    """The 67-parameter fixture of ``test_active_set_steps.py``: 3 linear Gaussians + 8x8 mesh."""
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
    tracer = al.Tracer(galaxies=[lens, source])
    return al, dataset, tracer


def test_positivity_off_routes_to_the_positive_negative_solver(tiny_fit):
    """``use_positive_only_solver=False`` is route (c), verified rather than asserted.

    Three facts in one: the positive-negative entry point is the one that runs,
    ``solve_ids_to_keep`` becomes ``None`` (so edge zeroing goes off with
    positivity — the trap the cell records beside every route-(c) timing), and
    the answer is the plain unconstrained solve of ``F + λH``.
    """
    al, dataset, tracer = tiny_fit

    with lsi.positive_negative_probe() as counts:
        fit = al.FitImaging(
            dataset=dataset,
            tracer=tracer,
            settings=al.Settings(use_border_relocator=True, use_positive_only_solver=False),
            xp=np,
        )
        reconstruction = np.asarray(fit.inversion.reconstruction, dtype=float)

    assert counts["calls"] >= 1, (
        "positivity off did not reach reconstruction_positive_negative_from"
    )
    assert fit.inversion.solve_ids_to_keep is None, "edge zeroing survived positivity being off"

    expected = np.linalg.solve(
        np.asarray(fit.inversion.curvature_reg_matrix, dtype=float),
        np.asarray(fit.inversion.data_vector, dtype=float),
    )
    assert np.allclose(reconstruction, expected, rtol=1e-9, atol=1e-12)
    assert int(np.sum(reconstruction < 0.0)) > 0, (
        "the fixture must actually produce negative pixels, or route (c) proves nothing"
    )


def test_positivity_on_keeps_edge_zeroing(tiny_fit):
    """The control: with positivity on, the solve IS subset to solve_ids_to_keep.

    This is why the injected wrapper's ``fixed0`` is ``zeros(n_keep)`` — the
    library has already removed the border pixels before the solver is called.
    """
    al, dataset, tracer = tiny_fit

    fit = al.FitImaging(
        dataset=dataset, tracer=tracer, settings=al.Settings(use_border_relocator=True), xp=np
    )
    ids = fit.inversion.solve_ids_to_keep
    assert ids is not None and len(np.asarray(ids)) < int(fit.inversion.total_params)
    reconstruction = np.asarray(fit.inversion.reconstruction, dtype=float)
    dropped = np.ones(int(fit.inversion.total_params), dtype=bool)
    dropped[np.asarray(ids, dtype=int)] = False
    assert np.all(reconstruction[dropped] == 0.0)


# ---------------------------------------------------------------------------
# 3. Static checks on the cell and its submits
# ---------------------------------------------------------------------------


def _submits() -> list[_Path]:
    return sorted(BATCH_GPU.glob("submit_breakdown_imaging_fixed_light_library_*"))


def test__the_cell_exists_and_declares_its_flags():
    assert CELL.is_file(), f"{CELL} is missing"
    text = CELL.read_text()
    for flag in ("--mesh", "--pass-budget", "--no-fallback-row", "--source-pixels"):
        assert flag in text, f"cell does not declare {flag}"


def test__the_cell_carries_phase_0s_certifying_budgets():
    """The default pass budget per mesh is phase 0's measurement, not a guess."""
    text = CELL.read_text()
    match = re.search(r"CERTIFYING_BUDGET = \{([^}]*)\}", text)
    assert match, "the cell does not declare CERTIFYING_BUDGET"
    body = match.group(1)
    for mesh, budget in (("rectangular", 7), ("delaunay", 2), ("delaunay_nn", 2)):
        assert f'"{mesh}": {budget}' in body, f"{mesh} budget is not phase 0's {budget}"


def test__three_fiducial_submits_exist():
    names = {p.name for p in _submits()}
    for mesh_label in ("pixelization", "delaunay", "delaunay_nn"):
        expected = f"submit_breakdown_imaging_fixed_light_library_{mesh_label}_a100_hst_fp64"
        assert expected in names, f"missing submit {expected}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_is_valid_bash(path):
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{path.name}: bash -n failed\n{result.stderr}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_runs_a_script_that_exists(path):
    text = path.read_text()
    scripts = re.findall(r"^python3 -u (\S+)", text, re.M)
    assert scripts, f"{path.name}: no `python3 -u <script>` line"
    for script in scripts:
        assert (ROOT / script).is_file(), f"{path.name}: runs missing script {script}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_passes_a_mesh_the_cell_accepts(path):
    text = path.read_text()
    meshes = re.findall(r"--mesh\s+(\S+)", text)
    assert meshes, f"{path.name}: the cell requires --mesh and the submit passes none"
    for mesh in meshes:
        assert mesh in MESHES, f"{path.name}: --mesh {mesh} is not one of {MESHES}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_writes_under_the_library_config_name(path):
    text = path.read_text()
    assert f"--config-name {CONFIG_NAME}" in text, (
        f"{path.name}: legs must write under --config-name {CONFIG_NAME} so the harvest "
        f"finds them, the phase-0 legs are not clobbered, and the dashboard does not "
        f"surface them"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_wipes_its_own_compilation_cache(path):
    # A seeded XLA autotune cache silently changes which kernels a leg uses
    # (results/notes/xla_autotune_triton_gemm.md: F at 4.8 vs 25.6 ms on the
    # same code). Every leg must start from an empty one.
    text = path.read_text()
    assert "JAX_COMPILATION_CACHE_DIR" in text, f"{path.name}: no per-job compilation cache"
    assert 'rm -rf "$JAX_COMPILATION_CACHE_DIR"' in text, (
        f"{path.name}: compilation cache directory is not wiped before the run"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_asks_for_the_batched_rows(path):
    text = path.read_text()
    assert "--vmap-batch 16" in text, f"{path.name}: the batched rows are part of the deliverable"


def test__launcher_exists_and_is_valid_bash():
    assert LAUNCHER.is_file(), f"{LAUNCHER} is missing"
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"{LAUNCHER.name}: bash -n failed\n{result.stderr}"


def test__launcher_lists_only_legs_that_exist():
    text = LAUNCHER.read_text()
    legs = re.findall(r"^submit_breakdown_imaging_fixed_light_library_\S+$", text, re.M)
    assert legs, f"{LAUNCHER.name} names no legs"
    for leg in legs:
        assert (BATCH_GPU / leg).is_file(), f"launcher names missing leg {leg}"


def test__launcher_covers_every_fiducial_leg():
    text = LAUNCHER.read_text()
    for mesh_label in ("pixelization", "delaunay", "delaunay_nn"):
        leg = f"submit_breakdown_imaging_fixed_light_library_{mesh_label}_a100_hst_fp64"
        assert leg in text, f"launcher does not submit {leg}"
