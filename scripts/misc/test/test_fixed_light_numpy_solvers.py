"""Unit tests for the jax-free numpy solver kernels and the solver cell (#265).

Three tiers:

1. **The kernel.** :func:`nnls_factor_reuse` must be the library's own answer —
   not near it — on systems whose unconstrained solution has a controlled number
   of negative entries, because "one factorisation instead of two" is only a
   speedup if the answer survives. The factorisation and downdate counters are
   pinned too: the whole claim of the kernel is that it spends **one**
   factorisation, and a counter that is merely reported can quietly stop being
   true.

2. **The jax-free contract.** ``fixed_light_cpu_common`` and
   ``fixed_light_numpy_solvers`` may not import JAX, ``active_set_steps``,
   ``library_solver_injection`` or ``fixed_light_cpu_kernels`` — at module level
   or inside a function. The numba cells' ``"jax" not in sys.modules`` assert is
   what that contract exists to protect, and it fires far from the import that
   broke it, so the import graph is checked here directly.

3. **Static checks on the solver cell.** It cannot be imported (module level runs
   the whole profile), so its constants are lifted out of its AST and the rest is
   asserted against the source: the thread pinning precedes the first ``import
   numpy``, there is no ``import jax``, the absent ``K2`` row is explained rather
   than merely missing, and ``resolve_output_paths`` is called with an explicit
   ``cell=``.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_numpy_solvers.py
"""

from __future__ import annotations

import ast
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

from likelihood_breakdown import fixed_light_cpu_common as flcc  # noqa: E402
from likelihood_breakdown import fixed_light_numpy_solvers as flns  # noqa: E402

CELL_PATH = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_numba_solvers.py"

MODULE_PATHS = (
    _misc / "likelihood_breakdown" / "fixed_light_cpu_common.py",
    _misc / "likelihood_breakdown" / "fixed_light_numpy_solvers.py",
)

#: The kernel must be the library's answer to this absolute tolerance on ``x``.
#: Tighter than the row's relative pin (1e-9), because on these small systems the
#: two solvers run the *same* arithmetic in the same order once the passive set
#: matches and disagree only by the seed's own round-off.
KERNEL_ATOL = 1.0e-10


# ---------------------------------------------------------------------------
# Fixtures: SPD systems with a controlled number of negative entries
# ---------------------------------------------------------------------------


def _spd_system(n, n_negative, seed):
    """An ``(n, n)`` SPD ``ZTZ`` and a ``ZTx`` whose unconstrained solve has signs.

    Built backwards from a chosen solution: ``ZTx = ZTZ @ x_target`` makes
    ``x_target`` exactly the unconstrained solution, so the number of entries the
    NNLS seed has to drop is chosen rather than discovered.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n + 12, n))
    ZTZ = A.T @ A + np.eye(n) * 0.5

    x_target = np.abs(rng.normal(size=n)) + 0.1
    if n_negative:
        x_target[rng.permutation(n)[:n_negative]] *= -1.0

    return ZTZ, ZTZ @ x_target, x_target


def _library_solve(ZTZ, ZTx):
    from autoarray.util.fnnls import fnnls_cholesky

    stats: dict = {}
    x = fnnls_cholesky(ZTZ, ZTx.T, P_initial=np.linalg.solve(ZTZ, ZTx) > 0, stats=stats)
    return x, stats


# ---------------------------------------------------------------------------
# 1. The kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "n_negative", "seed"), [(8, 3, 0), (25, 7, 1), (60, 19, 2)])
def test_factor_reuse_equals_the_library_fnnls(n, n_negative, seed):
    """Same solution, same passive set, same iteration counts as ``fnnls_cholesky``.

    The kernel differs from the library in exactly one place — where the factor
    it starts from comes from — so anything else differing is a bug, not a
    tolerance question. The passive sets are compared as sets: the library's
    ``P_inorder`` records insertion order, which the two need not share.
    """
    ZTZ, ZTx, _x_target = _spd_system(n, n_negative, seed)

    x_library, stats_library = _library_solve(ZTZ, ZTx)

    stats: dict = {}
    x_kernel = flns.nnls_factor_reuse(ZTZ, ZTx, stats=stats)

    assert np.max(np.abs(x_kernel - x_library)) <= KERNEL_ATOL
    assert np.all(x_kernel >= 0.0), "an NNLS solution with a negative entry is not one"
    assert set(stats["passive_set"].tolist()) == set(stats_library["passive_set"].tolist())
    assert stats["n_passive"] == stats_library["n_passive"]
    assert stats["outer_iterations"] == stats_library["outer_iterations"]


@pytest.mark.parametrize(("n", "n_negative", "seed"), [(8, 3, 0), (25, 7, 1), (60, 19, 2)])
def test_factor_reuse_spends_exactly_one_factorisation(n, n_negative, seed):
    """The whole claim of the kernel, as a counter rather than as prose.

    ``n_factorisations`` may only exceed one if the constraint fixer emptied the
    passive set entirely and the outer loop had to build a fresh 1x1 factor —
    which these systems do not do. Every other passive-set change is a downdate
    or an insertion into the factor that already exists.
    """
    ZTZ, ZTx, _ = _spd_system(n, n_negative, seed)

    stats: dict = {}
    flns.nnls_factor_reuse(ZTZ, ZTx, stats=stats)

    assert stats["n_factorisations"] == 1
    assert stats["seed_source"] == "factor_reuse_dense_sign"
    assert stats["n_seed_negatives"] == n_negative, (
        "the seed is the sign of the unconstrained solve, and the system was built "
        "backwards from a solution with exactly this many negative entries"
    )
    # Every seed negative leaves the passive set, and the repair loop may drop
    # further entries that only turn non-positive once their neighbours have gone.
    assert stats["n_downdates"] >= stats["n_seed_negatives"]


def test_factor_reuse_returns_the_unconstrained_solve_when_nothing_is_negative():
    """No negatives: one factorisation, no downdates, no iterations, and it says so."""
    ZTZ, ZTx, x_target = _spd_system(30, 0, seed=3)

    stats: dict = {}
    x = flns.nnls_factor_reuse(ZTZ, ZTx, stats=stats)

    assert np.max(np.abs(x - x_target)) <= 1e-8
    assert stats["n_factorisations"] == 1
    assert stats["n_downdates"] == 0
    assert stats["outer_iterations"] == 0
    assert stats["inner_iterations"] == 0
    assert stats["n_seed_negatives"] == 0
    assert stats["seed_source"] == "factor_reuse_unconstrained"
    assert stats["n_passive"] == 30

    x_library, _ = _library_solve(ZTZ, ZTx)
    assert np.max(np.abs(x - x_library)) <= KERNEL_ATOL


def test_library_nnls_solve_reports_the_two_factorisations_it_spends():
    """The reference row's cost model, recorded where a reader will see it.

    One LU for the sign seed (thrown away) plus one Cholesky of the passive
    block. Recording it beside the candidate's ``1`` is what makes the comparison
    legible without re-deriving it from the source.
    """
    ZTZ, ZTx, _ = _spd_system(20, 5, seed=4)

    stats: dict = {}
    flns.library_nnls_solve(ZTZ, ZTx, stats=stats)

    assert stats["n_factorisations"] == 2
    assert stats["seed_source"] == "dense"
    assert stats["warm_start_fallback"] is False


@pytest.mark.parametrize("new_signature", [True, False])
def test_numpy_injection_installs_against_the_old_and_new_library_signature(
    monkeypatch, new_signature
):
    """PyAutoArray #566 added ``solver``/``stats``; the injection must run on either side of it.

    The library passes ``solver=`` by name, so the wrapper must accept it on the numpy
    path (and still dispatch to the injected kernel), and on the delegated JAX path
    forward it only to a library that declares it.
    """
    import types

    from autoarray.inversion.inversion import inversion_util

    calls = []
    if new_signature:

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

    else:

        def stub(
            data_vector, curvature_reg_matrix, settings=None, xp=np, fingerprint=None, factor=None
        ):
            calls.append({})
            return data_vector

    monkeypatch.setattr(inversion_util, "reconstruction_positive_only_from", stub)

    kernel_calls = []

    def kernel(ZTZ, ZTx, *, stats):
        kernel_calls.append(1)
        return ZTx

    fake_jax = types.SimpleNamespace(__name__="jax.numpy")
    stats = {}
    with flns.numpy_solver_injected(kernel, label="stub") as counts:
        call = inversion_util.reconstruction_positive_only_from
        call(data_vector=np.ones(3), curvature_reg_matrix=np.eye(3), xp=np, solver="certified")
        call(
            data_vector=np.ones(3),
            curvature_reg_matrix=np.eye(3),
            xp=fake_jax,
            solver="certified",
            stats=stats,
        )
    assert (counts["numpy"], counts["jax"], len(kernel_calls)) == (1, 1, 1)
    if new_signature:
        assert calls == [{"solver": "certified", "stats": stats}]
    else:
        assert calls == [{}], "the old library must not be handed solver/stats"
    assert inversion_util.reconstruction_positive_only_from is stub


# ---------------------------------------------------------------------------
# 2. The jax-free contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda p: p.name)
def test_the_numpy_modules_import_no_jax_at_any_level(path):
    """Not at module level, not in a function, not behind a flag.

    A numba cell's ``"jax" not in sys.modules`` assert fires at the cell, far from
    the import that broke it. This check fires at the import.
    """
    banned_modules = {"jax", "jaxlib"}
    banned_names = {
        "active_set_steps",
        "library_solver_injection",
        "fixed_light_cpu_kernels",
        "reconstruction_steps",
    }

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                assert head not in banned_modules, f"{path.name}: `import {alias.name}`"
                assert alias.name.split(".")[-1] not in banned_names, (
                    f"{path.name}: `import {alias.name}`"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] not in banned_modules, (
                f"{path.name}: `from {module} import ...`"
            )
            assert module.split(".")[-1] not in banned_names, (
                f"{path.name}: `from {module} import ...`"
            )
            for alias in node.names:
                assert alias.name not in banned_names, (
                    f"{path.name}: `from {module} import {alias.name}`"
                )


def test_importing_the_numpy_solvers_leaves_a_fresh_process_jax_free():
    """The static check's runtime twin, in a process of its own.

    The import graph is what matters, and in *this* process another test has
    already imported JAX for its own reasons — so the only honest way to ask
    whether these modules drag it in is to ask a process that has nothing else in
    it. This is the assert the numba cells make about themselves, made here where
    the failure names the import rather than the cell.
    """
    import subprocess
    import sys

    script = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(_misc)!r})",
            "from likelihood_breakdown import fixed_light_cpu_common",
            "from likelihood_breakdown import fixed_light_numpy_solvers",
            "leaked = sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib'))",
            "print('LEAKED:' + ','.join(leaked))",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "LEAKED:\n" in result.stdout or result.stdout.strip().endswith("LEAKED:"), (
        f"importing the numpy solver modules pulled JAX into a fresh process: "
        f"{result.stdout.strip()}"
    )


def test_the_cpu_kernels_module_shares_these_helpers_rather_than_copying_them():
    """One timer, one thread block, one solver view — the same objects, not twins.

    A row measured through ``fixed_light_cpu_kernels`` and a row measured through
    ``fixed_light_numpy_solvers`` must be timed by the same function, or the two
    modules' milliseconds are only comparable by inspection.
    """
    from likelihood_breakdown import fixed_light_cpu_kernels as flck

    assert flck.time_median is flcc.time_median
    assert flck.thread_block is flcc.thread_block
    assert flck.solver_view_of is flcc.solver_view_of
    assert flck.BLAS_THREAD_VARS is flcc.BLAS_THREAD_VARS


# ---------------------------------------------------------------------------
# 3. The numpy evidence scorer
# ---------------------------------------------------------------------------


def test_log_evidence_terms_np_is_the_jax_scorers_arithmetic():
    """Term for term the JAX scorer, in numpy — pinned against it, not asserted.

    The numpy twin exists because a numba cell may not import JAX to score a row
    it measured without one. That makes it a re-implementation, and a
    re-implementation that is never compared against its original is a second
    definition waiting to drift.
    """
    pytest.importorskip("jax")
    autolens = pytest.importorskip("autolens")
    al = autolens

    from likelihood_breakdown import fixed_light_system as fls

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
    system = fls.fixed_light_system_from(
        fit, dataset, name="S3", scaling="numpy", sparse_operator="drop"
    )

    x = np.asarray(system.inversion.reconstruction, dtype=float)

    numpy_terms = flcc.log_evidence_terms_np(system, x)
    jax_terms = system.log_evidence_terms(x)

    for key, value in jax_terms.items():
        assert numpy_terms[key] == pytest.approx(value, rel=1e-10, abs=1e-10), key


# ---------------------------------------------------------------------------
# 4. Static checks on the solver cell
# ---------------------------------------------------------------------------

_LIFTED_CONSTANTS = (
    "ALL_KERNEL_KEYS",
    "ALL_STREAM_KEYS",
    "IID_SEED",
    "WALK_SEED",
    "RANDOM_WALK_STEP_FRACTION",
    "IID_UNIT_LOW",
    "IID_UNIT_HIGH",
)


@pytest.fixture(scope="module")
def cell_ns() -> dict:
    """The cell's constants, lifted from its AST — it cannot be imported."""
    namespace: dict = {}
    tree = ast.parse(CELL_PATH.read_text())
    wanted: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in _LIFTED_CONSTANTS for name in targets):
                wanted.append(node)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(CELL_PATH), "exec"), namespace)
    return namespace


def test_the_solver_cell_declares_its_rows_and_explains_the_absent_one(cell_ns):
    """``K2`` is missing on purpose, and the JSON says why in a field of its own.

    A row that is simply absent reads as an oversight. This one is a decision —
    the numpy certified active set lives behind a JAX import and this process may
    not have one — and the artifact has to carry the reason, not the note.
    """
    assert cell_ns["ALL_KERNEL_KEYS"] == ("K0", "K1", "K1i", "K3", "K4")
    assert "K2" not in cell_ns["ALL_KERNEL_KEYS"]
    assert cell_ns["ALL_STREAM_KEYS"] == ("walk", "iid")

    source = CELL_PATH.read_text()
    assert '"absent_row_K2"' in source
    flat = " ".join(source.split())
    assert "There is no ``K2``" in flat
    assert "active_set_steps" in flat


def test_the_solver_cell_never_imports_jax():
    """Not at module level, not in a function, not behind a flag."""
    tree = ast.parse(CELL_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "jax", f"`import {alias.name}`"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] != "jax", f"`from {module} import ...`"

    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }
    assert "device_info_dict" not in imported, (
        "device_info_dict imports jax unconditionally (_profile_cli.py:392)"
    )

    # ...and the process asserts it for itself, at import.
    source = CELL_PATH.read_text()
    assert source.count('assert "jax" not in sys.modules') == 1


def test_the_cell_records_rather_than_asserts_the_post_run_jax_state():
    """The library imports JAX for itself, so the after-the-rows state is DATA.

    ``AbstractNDArray.__getitem__`` runs ``import jax.numpy as jnp`` on every
    index, guarded only by ``ImportError`` (``abstract_ndarray.py:422-428``), and
    the first ``FitImaging`` reaches it through ``BorderRelocator``. In an
    environment where JAX is installed, no process that builds a fit can stay
    JAX-free — so an assert there would fail on every run for a reason the cell
    cannot fix, and would say nothing about the cell. What the import-time assert
    buys survives and is recorded beside it: no XLA device, no JAX computation,
    ``NPROC`` never consulted.
    """
    source = CELL_PATH.read_text()

    assert '"jax_after_rows": jax_after_rows' in source, (
        "the post-row JAX state must reach the artifact"
    )
    assert '"asserted_after_rows": False' in source
    assert "abstract_ndarray.py:422-428" in source
    assert "BorderRelocator" in source

    # And the claim is checkable: the library really does import jax there.
    import autoarray

    getitem_source = (_Path(autoarray.__file__).parent / "abstract_ndarray.py").read_text()
    assert "import jax.numpy as jnp" in getitem_source, (
        "the note in the cell describes a library import that no longer exists; "
        "re-check whether the post-row assert can come back"
    )


def test_thread_pinning_precedes_the_first_numpy_import():
    """OpenBLAS reads its thread count once, when the shared library loads.

    A row timed at ``--threads 1`` in a process whose BLAS was already up with
    eight is not a single-threaded row, and nothing in the output would say so.
    """
    source = CELL_PATH.read_text()

    pin = source.index("thread_env = _pin_thread_env(N_THREADS)")
    numba = source.index('_os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)')
    numpy_import = source.index("import numpy as np")

    assert pin < numpy_import, "BLAS pinned after numpy was imported"
    assert numba < numpy_import, "NUMBA_NUM_THREADS set after numpy was imported"


def test_the_solver_cell_resolves_its_own_output_paths_with_an_explicit_cell():
    """Without ``cell=`` the first-token rule derives ``fixed`` and clobbers phase 0."""
    tree = ast.parse(CELL_PATH.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_output_paths"
    ]
    assert len(calls) == 1
    assert "cell" in {kw.arg for kw in calls[0].keywords}


def test_the_solver_cell_declares_its_own_timing_status():
    """A smoke leg's milliseconds are wiring evidence and must say so in the JSON."""
    source = CELL_PATH.read_text()

    assert '"timing_status": _timing_status' in source
    assert '"wiring_only_not_measured"' in source
    assert '"measured_under_contention"' in source
    assert "/proc/loadavg" in source
    assert '"smoke" in _cli.config_name.lower()' in source


def test_the_memo_rows_are_one_call_per_system_not_n_repeats_on_one():
    """Repeating one solve warm-starts it from its own answer — a solve nobody gets."""
    source = (_misc / "likelihood_breakdown" / "fixed_light_numpy_solvers.py").read_text()
    tree = ast.parse(source)
    memo = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "memo_warm_row"
    )
    body = ast.unparse(memo)

    parameters = {arg.arg for arg in memo.args.args} | {arg.arg for arg in memo.args.kwonlyargs}
    assert "n_repeats" not in parameters, (
        "memo_warm_row must not take an n_repeats loop over one system — the stream "
        "length is the sample size, and repeating one solve warm-starts it from its "
        "own answer"
    )
    assert "for index, one_system in enumerate(systems)" in body
    # The env is set for the row and restored, and the memo is cleared either side.
    # `ast.unparse` normalises quoting, so match its own rendering.
    assert "os.environ['AUTOARRAY_NNLS_WARM_START'] = '1'" in body
    assert "_nnls_passive_set_memo.clear()" in body
    assert "finally:" in body
