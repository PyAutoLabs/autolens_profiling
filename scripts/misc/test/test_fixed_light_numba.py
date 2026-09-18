"""Unit tests for the numba CPU fixed-lens-light cell (autolens_profiling#263).

Two tiers, on the 30x30 / 8x8 / 3-linear-Gaussian ``FitImaging`` fixture
``test_active_set_steps.py`` and ``test_fixed_light_cpu_kernels.py`` already
build — the same geometry, so a failure here is a failure of *this* cell rather
than of the fixture.

1. **The library claims the cell rests on.** The factory really does dispatch a
   different inversion class per formalism (``P1``/dispatch); the operator
   re-baked on the subtracted dataset really is the original's (``P1``); the two
   formalisms really do assemble the same linear system (``P2``) and solve it to
   the same evidence (``P3``); ``psf_weighted_data`` really does track the
   dataset's data rather than a baked weight map — the *positive* statement that
   makes a sparse S3 row legitimate, which no amount of prose can substitute
   for. And ``call_accounting`` over a real likelihood call really does cover
   95 % of it at under 3 % overhead, which is the whole premise of the
   decomposition.

   ``curvature_reg_matrix`` is pinned at ``n_calls >= 2`` deliberately. It is a
   plain ``property`` (``abstract.py:358``) and the library recomputes it several
   times per call; if a future PyAutoArray change caches it, the decomposition's
   ``F + lambda H`` row silently stops meaning what the note says it means. This
   test makes that change fail loudly so the row set is updated on purpose.

2. **Static checks on the cell.** The cell cannot be imported (module level runs
   the whole profile), so its helpers and constants are lifted out of its AST and
   executed here, and the rest is asserted against the source: the thread pinning
   precedes the first ``import numpy``, there is no ``import jax`` anywhere,
   ``ALL_ROUTE_KEYS`` is exactly ``("a", "b", "c")``, ``use_jax=False`` reaches
   ``AnalysisImaging``, and ``resolve_output_paths`` is called with an explicit
   ``cell=`` (without which the first-token rule derives ``fixed`` and clobbers
   the phase-0 ``fixed_light*`` artifacts — ``_profile_cli.py:558-562``).

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_numba.py
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

from likelihood_breakdown import call_accounting as ca  # noqa: E402
from likelihood_breakdown import fixed_light_numpy_solvers as flns  # noqa: E402
from likelihood_breakdown import fixed_light_system as fls  # noqa: E402

CELL_PATH = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_numba.py"

P2_RTOL = 1.0e-9
P3_RTOL = 1.0e-6
P4_RTOL = 1.0e-9


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    """The 67-parameter S0 fit of ``test_active_set_steps.py``, plus its dataset."""
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
def tiny_s3_pair(tiny):
    """The dense and sparse-numba S3 systems built from the same S0 fit."""
    fit, dataset = tiny
    dense = fls.fixed_light_system_from(
        fit, dataset, name="S3_dense", scaling="numpy", sparse_operator="drop"
    )
    sparse = fls.fixed_light_system_from(
        fit, dataset, name="S3_sparse", scaling="numpy", sparse_operator="rebake_cpu"
    )
    return dense, sparse


def _clear_nnls_memo():
    from autoarray.inversion.inversion import nnls_memo

    nnls_memo._nnls_passive_set_memo.clear()


# ---------------------------------------------------------------------------
# 1. The library claims
# ---------------------------------------------------------------------------


def test_the_factory_dispatches_a_different_class_per_formalism(tiny_s3_pair):
    """A structural fact, not an inference from a flag (``factory.py:136-154``)."""
    from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping
    from autoarray.inversion.inversion.imaging_numba.sparse import (
        InversionImagingSparseNumba,
    )

    dense, sparse = tiny_s3_pair

    assert isinstance(dense.inversion, InversionImagingMapping)
    assert isinstance(sparse.inversion, InversionImagingSparseNumba)
    assert dense.dataset.sparse_operator is None
    assert sparse.dataset.sparse_operator is not None
    assert dense.fit._xp is np
    assert sparse.fit._xp is np


def test_p1_the_rebaked_operator_is_the_original(tiny):
    """P1: the operator is a function of the noise map, PSF and mask — not the data.

    Asserted element for element on all three arrays, because everything the
    sparse S3 rows claim rests on it.
    """
    fit, dataset = tiny

    with_operator = dataset.apply_sparse_operator_cpu()
    rebaked = fls.fixed_light_system_from(
        fit, dataset, name="S3", scaling="numpy", sparse_operator="rebake_cpu"
    ).dataset

    for field in ("psf_precision_operator_sparse", "indexes", "lengths"):
        assert np.array_equal(
            np.asarray(getattr(with_operator.sparse_operator, field)),
            np.asarray(getattr(rebaked.sparse_operator, field)),
        ), f"{field} differs between the original and the re-baked operator"


def test_p2_the_two_formalisms_assemble_the_same_system(tiny_s3_pair):
    """P2: ``D`` and the mapper block of ``F`` agree to 1e-9.

    This is the statement that the subtraction reached the *sparse* data vector.
    The JAX sparse class reads a weight map baked from the unsubtracted image and
    would fail here; the numba one recomputes from ``self.data`` and does not.
    """
    dense, sparse = tiny_s3_pair

    dv_dense = np.asarray(dense.inversion.data_vector, dtype=float)
    dv_sparse = np.asarray(sparse.inversion.data_vector, dtype=float)

    n_mapper = int(dense.n_mapper)
    cm_dense = np.asarray(dense.inversion.curvature_matrix, dtype=float)[:n_mapper, :n_mapper]
    cm_sparse = np.asarray(sparse.inversion.curvature_matrix, dtype=float)[:n_mapper, :n_mapper]

    def max_rel(got, ref):
        scale = max(float(np.max(np.abs(ref))), 1e-300)
        return float(np.max(np.abs(got - ref))) / scale

    assert max_rel(dv_sparse, dv_dense) <= P2_RTOL
    assert max_rel(cm_sparse, cm_dense) <= P2_RTOL


def _p3_measure(tiny_s3_pair) -> dict:
    """Measure P3 and RETURN it: the Δ in nats, the relative difference, the
    number of passive-set differences between the two solves.

    A helper rather than the test body, because the numbers are the point: an
    assert that only says "close enough" hides a formalism drifting towards the
    tolerance, and the cell reports all three per leg.
    """
    from autoarray.util import fnnls as fnnls_module

    dense, sparse = tiny_s3_pair

    stats = {}
    for key, system in (("dense", dense), ("sparse", sparse)):
        _clear_nnls_memo()
        ca.install([ca.function_site("fnnls", fnnls_module, "fnnls_cholesky")])
        try:
            fom = float(system.fit.figure_of_merit)
            stats[key] = (fom, ca.captured_stats().get("fnnls", {}))
        finally:
            ca.uninstall()

    fom_dense, stats_dense = stats["dense"]
    fom_sparse, stats_sparse = stats["sparse"]

    d_nats = fom_sparse - fom_dense
    rel = abs(d_nats) / max(abs(fom_dense), 1e-300)

    ps_dense = np.sort(np.asarray(stats_dense["passive_set"], dtype=int))
    ps_sparse = np.sort(np.asarray(stats_sparse["passive_set"], dtype=int))
    n_passive_set_differences = int(len(np.setxor1d(ps_dense, ps_sparse)))

    return {
        "figure_of_merit_dense": fom_dense,
        "figure_of_merit_sparse_numba": fom_sparse,
        "d_log_evidence_nats": d_nats,
        "rel_diff": rel,
        "n_passive_set_differences": n_passive_set_differences,
        "n_passive_dense": int(ps_dense.size),
        "n_passive_sparse_numba": int(ps_sparse.size),
    }


def test_p3_the_two_formalisms_solve_to_the_same_evidence(tiny_s3_pair):
    """P3: the same figure of merit to 1e-6, with the measured Δ printed beside it."""
    measured = _p3_measure(tiny_s3_pair)
    print(f"  P3: {measured}")

    assert measured["rel_diff"] <= P3_RTOL, (
        f"dense {measured['figure_of_merit_dense']!r} vs sparse-numba "
        f"{measured['figure_of_merit_sparse_numba']!r}: "
        f"Δ {measured['d_log_evidence_nats']:+.6e} nats "
        f"(rel {measured['rel_diff']:.3e}), "
        f"{measured['n_passive_set_differences']} passive-set differences"
    )
    assert measured["n_passive_dense"] > 0, "the fixture's NNLS solve has no passive set"


def test_psf_weighted_data_tracks_the_datasets_data(tiny, tiny_s3_pair):
    """The POSITIVE statement: the numba path reads no baked weight map.

    ``InversionImagingSparseNumba.psf_weighted_data`` recomputes from
    ``self.data`` on every inversion (``imaging_numba/sparse.py:94-101``). The S0
    and S3 sparse inversions carry an *identical* operator (P1) but different
    data, so if the property read the operator's baked map the two would be
    equal — which is exactly the failure mode that confined the JAX legs of this
    epic to the dense path.
    """
    fit, dataset = tiny
    _dense, sparse_s3 = tiny_s3_pair

    with_operator = dataset.apply_sparse_operator_cpu()
    fit_s0_sparse = fit.__class__(
        dataset=with_operator,
        tracer=fit.tracer,
        settings=fit.settings,
        xp=np,
    )

    weighted_s0 = np.asarray(fit_s0_sparse.inversion.psf_weighted_data, dtype=float)
    weighted_s3 = np.asarray(sparse_s3.inversion.psf_weighted_data, dtype=float)

    assert weighted_s0.shape == weighted_s3.shape
    assert not np.allclose(weighted_s0, weighted_s3), (
        "psf_weighted_data is identical on the subtracted and unsubtracted datasets — "
        "the numba path is reading a baked weight map and the sparse S3 rows are wrong."
    )

    # ...and the difference is the subtraction, not noise: the subtracted light
    # carries real flux.
    assert abs(float(sparse_s3.subtracted_light_flux)) > 0.0


def test_the_three_sparse_operator_modes(tiny):
    """``drop`` / ``carry`` / ``rebake_cpu`` do three distinct, named things."""
    from autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util import (
        SparseLinAlgImagingNumba,
    )

    fit, dataset = tiny
    with_operator = dataset.apply_sparse_operator_cpu()
    fit_with_operator = fit.__class__(
        dataset=with_operator,
        tracer=fit.tracer,
        settings=fit.settings,
        xp=np,
    )

    dropped = fls.fixed_light_system_from(
        fit_with_operator, with_operator, name="drop", scaling="numpy", sparse_operator="drop"
    )
    carried = fls.fixed_light_system_from(
        fit_with_operator, with_operator, name="carry", scaling="numpy", sparse_operator="carry"
    )
    rebaked = fls.fixed_light_system_from(
        fit_with_operator,
        with_operator,
        name="rebake",
        scaling="numpy",
        sparse_operator="rebake_cpu",
    )

    assert dropped.dataset.sparse_operator is None
    assert carried.dataset.sparse_operator is with_operator.sparse_operator
    assert isinstance(rebaked.dataset.sparse_operator, SparseLinAlgImagingNumba)
    assert rebaked.dataset.sparse_operator is not with_operator.sparse_operator


def test_the_rebuild_carries_the_psf_over_sampling_and_noise_covariance(tiny):
    """The fields the previous rebuild silently dropped (``dataset.py:65-79``).

    ``apply_sparse_operator`` REFUSES an over-sampled PSF
    (``dataset.py:622-627``); defaulting the field back to 1 would turn that
    refusal into a quietly different dataset.
    """
    fit, dataset = tiny
    system = fls.fixed_light_system_from(fit, dataset, name="S3", scaling="numpy")

    assert system.dataset.convolve_over_sample_size_lp == dataset.convolve_over_sample_size_lp
    assert (
        system.dataset.convolve_over_sample_size_pixelization
        == dataset.convolve_over_sample_size_pixelization
    )
    if dataset.noise_covariance_matrix is None:
        assert system.dataset.noise_covariance_matrix is None
    else:
        assert np.array_equal(
            np.asarray(system.dataset.noise_covariance_matrix),
            np.asarray(dataset.noise_covariance_matrix),
        )


# ---------------------------------------------------------------------------
# The cell's own site spec, lifted from its AST
# ---------------------------------------------------------------------------

_LIFTED_FUNCTIONS = ("_site_spec", "_parse_routes", "_parse_formalisms")
_LIFTED_CONSTANTS = (
    "ALL_ROUTE_KEYS",
    "DEFAULT_ROUTE_KEYS",
    "ALL_FORMALISM_KEYS",
    "GROUP_LABELS",
    "MAX_INSTRUMENTATION_OVERHEAD_MS",
    "REFERENCE_OVERHEAD_RATIO",
    "MIN_BLOCKS_FOR_OVERHEAD_ASSERT",
    "MAX_UNATTRIBUTED_FRACTION",
    "WARMUP_WINDOW",
    "WARMUP_TOLERANCE",
    "WARMUP_MAX_CALLS",
    "UNATTRIBUTED_LABEL",
    # Lever 3 of #267: the site spec reads whether the installed PyAutoArray caches
    # `curvature_reg_matrix` off the class, so the lifted spec needs the same value.
    "CURVATURE_REG_MATRIX_IS_CACHED",
    "P2_RTOL",
    "P3_RTOL",
    "P4_RTOL",
    "P5_RTOL",
    "MAPPER_LOGDET_RTOL",
)


def _cell_namespace() -> dict:
    """Execute the cell's helpers + constants, and nothing else, in a fresh dict.

    The names the lifted ``_site_spec`` closes over are the library classes the
    cell imports; they are supplied here so the spec under test is the *real*
    one. A site the cell adds without this test seeing it is a site whose
    ``cached`` declaration is never checked.
    """
    import functools
    import inspect as _inspect

    from autoarray.inversion.inversion import inversion_util
    from autoarray.inversion.inversion.abstract import AbstractInversion
    from autoarray.inversion.inversion.imaging.abstract import AbstractInversionImaging
    from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping
    from autoarray.inversion.inversion.imaging_numba.sparse import (
        InversionImagingSparseNumba,
    )
    from autoarray.inversion.mappers.abstract import Mapper
    from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
    from autoarray.util import fnnls as fnnls_module
    from autolens.imaging.fit_imaging import FitImaging
    from autolens.lens.to_inversion import TracerToInversion
    from autonerves import cached_property as autonerves_cached_property

    namespace = {
        # `CURVATURE_REG_MATRIX_IS_CACHED`'s lifted assignment evaluates
        # `isinstance(inspect.getattr_static(...), (functools.cached_property,
        # autonerves_cached_property))`, so those three names have to be here too.
        "functools": functools,
        "inspect": _inspect,
        "autonerves_cached_property": autonerves_cached_property,
        "call_accounting": ca,
        "inversion_util": inversion_util,
        "fnnls_module": fnnls_module,
        "fixed_light_numpy_solvers": flns,
        "AbstractInversion": AbstractInversion,
        "AbstractInversionImaging": AbstractInversionImaging,
        "InversionImagingMapping": InversionImagingMapping,
        "InversionImagingSparseNumba": InversionImagingSparseNumba,
        "Mapper": Mapper,
        "InterpolatorDelaunay": InterpolatorDelaunay,
        "FitImaging": FitImaging,
        "TracerToInversion": TracerToInversion,
    }

    tree = ast.parse(CELL_PATH.read_text())
    wanted: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _LIFTED_FUNCTIONS:
            wanted.append(node)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in _LIFTED_CONSTANTS for name in targets):
                wanted.append(node)

    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(CELL_PATH), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def cell_ns():
    return _cell_namespace()


def test_call_accounting_covers_a_real_likelihood_call(tiny_s3_pair, cell_ns):
    """The cell's own spec, over a real fit: >= 95 % attributed at <= 3 % overhead.

    Both halves matter. Coverage below 95 % means the spec does not describe the
    call graph and the decomposition is mostly remainder; overhead above 3 %
    means the instrument is changing the number it reports.

    The overhead is measured **counterbalanced** (A B B A per block, three
    blocks), exactly as the cell measures it. A single clean call against a
    single instrumented call is the estimator that produced 0.71 to 1.37 on the
    cell's own rows; run that way here it returns ~1.13 on this fixture, which is
    the fixture's scatter and not the harness.
    """
    import time

    _dense, sparse = tiny_s3_pair
    fit = sparse.fit

    def one_call():
        _clear_nnls_memo()
        fresh = fit.__class__(
            dataset=fit.dataset,
            tracer=fit.tracer,
            adapt_images=fit.adapt_images,
            settings=fit.settings,
            xp=np,
        )
        started = time.perf_counter()
        value = float(fresh.figure_of_merit)
        return time.perf_counter() - started, value

    one_call()  # warm-up: numba compiles here

    block_ratios = []
    values = []
    snapshot = None
    instrumented_s = None
    for _block in range(3):
        a1, value_a1 = one_call()
        ca.install(cell_ns["_site_spec"]())
        try:
            b1, value_b1 = one_call()
            b2, value_b2 = one_call()
            snapshot = ca.snapshot()
        finally:
            ca.uninstall()
        a2, value_a2 = one_call()
        block_ratios.append(((b1 + b2) / 2.0) / ((a1 + a2) / 2.0))
        instrumented_s = (b1 + b2) / 2.0
        values.extend([value_a1, value_b1, value_b2, value_a2])

    # The instrumentation changes the timing, never the answer.
    for value in values:
        assert value == pytest.approx(values[0], rel=1e-12)

    # Two instrumented calls per block, so the snapshot covers two calls.
    attributed = sum(entry["excl_s"] for entry in snapshot.values()) / 2.0
    coverage = attributed / instrumented_s
    overhead = float(np.mean(block_ratios))
    print(f"  coverage {coverage * 100:.2f} %, ABBA overhead x{overhead:.4f} {block_ratios}")

    assert coverage >= 0.95, (
        f"call_accounting attributed only {coverage * 100:.2f} % of the call "
        f"({attributed * 1e3:.3f} ms of {instrumented_s * 1e3:.3f} ms); the site spec "
        f"does not describe this call graph."
    )
    # The overhead verdict is repeat-conditional, exactly as the cell's gate is: a
    # measurement whose own block-to-block spread exceeds the threshold cannot
    # resolve the threshold, and does not get to render a verdict. (Measured on a
    # contended host: blocks 0.94-1.15 around a mean of 1.02, against a 1.03
    # threshold.) A GROSS regression still fails, because a harness that doubled
    # the call would clear any noise floor.
    # This fixture's call is a few milliseconds, so the cell's own MILLISECOND
    # budget (12 ms, phase 4) cannot discriminate anything here: every ratio
    # short of catastrophic clears it. The reference RATIO is the right gate at
    # this scale, and it is the one this test has always used.
    threshold = cell_ns["REFERENCE_OVERHEAD_RATIO"]
    block_spread = (max(block_ratios) - min(block_ratios)) / overhead
    if block_spread <= threshold - 1.0:
        assert overhead <= threshold, (
            f"ABBA instrumentation overhead x{overhead:.4f} exceeds {threshold} over "
            f"blocks {block_ratios}"
        )
    else:
        print(
            f"  overhead RECORDED, not asserted: block spread {block_spread * 100:.1f} % "
            f"cannot resolve a {(threshold - 1.0) * 100:.0f} % threshold on this host."
        )
        assert overhead <= 1.5, (
            f"ABBA instrumentation overhead x{overhead:.4f} is gross even against a "
            f"{block_spread * 100:.1f} % noise floor; blocks {block_ratios}"
        )

    # Every site declared cached was reached at most once per call.
    for label in ca.declared_cached_labels():
        assert snapshot[label]["n_calls"] <= 2, (
            f"{label} is declared cached but was reached "
            f"{snapshot[label]['n_calls']} times in two calls."
        )


def test_curvature_reg_matrix_is_cached(tiny_s3_pair, cell_ns):
    """``F + lambda H`` is a ``cached_property`` and the library builds it once per call.

    Pins the cache. It became a ``cached_property`` in lever 3 of #267 (PyAutoArray
    ``b4322c3e``), having been a plain ``property`` the likelihood rebuilt on every
    access — twice per evaluation. If a future PyAutoArray change stops caching it,
    this fails and the ``F + lambda H`` row of the decomposition is updated on
    purpose rather than quietly becoming a different quantity.
    """
    _dense, sparse = tiny_s3_pair
    fit = sparse.fit

    _clear_nnls_memo()
    fresh = fit.__class__(
        dataset=fit.dataset,
        tracer=fit.tracer,
        adapt_images=fit.adapt_images,
        settings=fit.settings,
        xp=np,
    )

    ca.install(cell_ns["_site_spec"]())
    try:
        float(fresh.figure_of_merit)
        snapshot = ca.snapshot()
    finally:
        ca.uninstall()

    assert snapshot["inversion.curvature_reg_matrix"]["n_calls"] == 1, (
        "AbstractInversion.curvature_reg_matrix was reached "
        f"{snapshot['inversion.curvature_reg_matrix']['n_calls']} time(s), not once. If "
        "PyAutoArray no longer caches it, update the decomposition's F + lambda H row on "
        "purpose."
    )


# ---------------------------------------------------------------------------
# 2. Static checks on the cell
# ---------------------------------------------------------------------------


def test_thread_pinning_precedes_the_first_numpy_import():
    """OpenBLAS/MKL and numba read their knobs once, when their libraries load."""
    source = CELL_PATH.read_text()
    pin_blas = source.index("_pin_thread_env(N_THREADS)")
    pin_numba = source.index('_os.environ["NUMBA_NUM_THREADS"]')
    first_numpy = source.index("import numpy as np")

    assert pin_blas < first_numpy, "pin_thread_env must run before `import numpy`"
    assert pin_numba < first_numpy, "NUMBA_NUM_THREADS must be set before `import numpy`"


def test_the_cell_never_imports_jax():
    """Not at module level, not in a function, not behind a flag."""
    tree = ast.parse(CELL_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "jax", f"`import {alias.name}`"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] != "jax", f"`from {module} import ...`"

    # ...and `device_info_dict` is never imported or called: it imports jax
    # unconditionally (`_profile_cli.py:392`). The cell names it in a comment
    # explaining why it is absent, which is why this is an AST check and not a
    # substring search.
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }
    assert "device_info_dict" not in imported

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "device_info_dict" not in called, (
        "device_info_dict imports jax unconditionally (_profile_cli.py:392) and must "
        "not be called by this cell."
    )


def test_all_route_keys_is_a_b_the_two_kernel_routes_c_and_d_np(cell_ns):
    """Six routes exist; only three run by default.

    ``d_np`` patches the library's positive-only entry point, and ``b_direct`` /
    ``b_touched`` (phase 4, #274) patch its curvature-matrix dispatcher, each for
    the duration of their own row. A cell invocation written before any of them
    existed must produce the row set it always did, so ``--routes`` defaults to
    :data:`DEFAULT_ROUTE_KEYS`, not to :data:`ALL_ROUTE_KEYS`.

    ``b_direct`` and ``b_touched`` sit immediately after ``b`` in the canonical
    order because the A/B is read as one table: ``b`` (production, two-stage) on
    top, the two candidate kernels under it.

    Still no ``d``/``d0``/``e``: those are the JAX certified-active-set rows of
    another cell, and this one imports no JAX.
    """
    assert cell_ns["ALL_ROUTE_KEYS"] == ("a", "b", "b_direct", "b_touched", "c", "d_np", "d_perm")
    assert cell_ns["DEFAULT_ROUTE_KEYS"] == ("a", "b", "c")
    assert cell_ns["_parse_routes"](None) == ("a", "b", "c"), (
        "the default row set must not silently grow d_np, b_direct or b_touched"
    )
    assert cell_ns["_parse_routes"]("c,a") == ("a", "c")
    assert cell_ns["_parse_routes"]("b,d_np") == ("b", "d_np")
    assert cell_ns["_parse_routes"]("d_np") == ("d_np",)
    assert cell_ns["_parse_routes"]("b,d_perm") == ("b", "d_perm")
    # The submit's own route string, in the order the summary will print it.
    assert cell_ns["_parse_routes"]("b,b_direct,b_touched") == ("b", "b_direct", "b_touched")
    assert cell_ns["_parse_routes"]("b_touched,b") == ("b", "b_touched")
    with pytest.raises(ValueError, match="unknown route"):
        cell_ns["_parse_routes"]("a,d")
    with pytest.raises(ValueError, match="unknown route"):
        cell_ns["_parse_routes"]("b_two_stage")
    assert cell_ns["ALL_FORMALISM_KEYS"] == ("dense", "sparse_numba")
    assert cell_ns["_parse_formalisms"]("both") == ("dense", "sparse_numba")
    assert cell_ns["_parse_formalisms"]("dense") == ("dense",)


def test_route_d_np_is_route_b_with_one_function_replaced():
    """Every table the row loop reads must treat ``d_np`` exactly as it treats ``b``.

    A ``d_np`` row on a different dataset, different instances or different
    settings would make the ``b -> d_np`` delta a comparison of two problems
    rather than of two solvers. The tables are read out of the cell's AST because
    they are built from module-level objects this test cannot construct.
    """
    source = CELL_PATH.read_text()

    for route in ("d_np", "d_perm", "b_direct", "b_touched"):
        assert f'("{route}", "dense"): dataset_s3_dense,' in source
        assert f'("{route}", "sparse_numba"): dataset_s3_sparse,' in source
    # The two branches that split a route off from b's setup name only "a" and
    # "c", so every _ROUTES_LIKE_B route falls through to b's adapt images,
    # instances and settings.
    assert 'adapt_images if route == "a" else adapt_images_s3' in source
    assert 'settings=_settings if route != "c" else _settings_positive_negative' in source
    assert 'instances[index] if route == "a" else instances_s3[index]' in source
    assert '"d_np"' in source and "_ROUTES_LIKE_B" in source
    assert '_ROUTES_LIKE_B = ("b", "d_np", "d_perm", "b_direct", "b_touched")' in source


def test_the_injection_is_entered_outside_call_accounting(cell_ns):
    """The row's patch must be in place before ``install`` wraps the dotted name.

    ``call_accounting.function_site`` rebinds
    ``inversion_util.reconstruction_positive_only_from`` too. Entered outside, the
    injection is in place first and the accounting wrapper closes over the
    injected function, so the decomposition attributes the injected solve.
    Entered inside, ``uninstall()`` would restore the library function over the
    injection and the row would measure the wrong kernel.
    """
    source = CELL_PATH.read_text()
    tree = ast.parse(source)

    # `install` is only ever called inside `_abba_blocks` / `_solve_with_stats`,
    # and the injection is entered in the row loop, which is module level.
    assert "_injection.__enter__()" in source
    assert "_injection.__exit__(None, None, None)" in source
    assert source.index("_injection_counts = _injection.__enter__()") < source.index(
        "_inversion_class = _assert_dispatch(_route, _formalism)"
    ), "the injection must be entered before the row does anything"

    # And the counters are asserted, not merely recorded: a patch that never
    # fired would report route b's timing wearing route d_np's label.
    assert 'if _injection_counts["numpy"] <= 0:' in source
    assert "wearing route d_np's label" in source

    # The kernel is reached through the module attribute, so `install` can wrap it.
    assert "_factor_reuse_solver" in source
    factor = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_factor_reuse_solver"
    )
    assert "fixed_light_numpy_solvers.nnls_factor_reuse" in ast.unparse(factor), (
        "the kernel must be resolved on the module at call time, or call_accounting cannot wrap it"
    )
    assert cell_ns["P4_RTOL"] == 1.0e-9
    assert cell_ns["P5_RTOL"] == 1.0e-9
    assert "P5_d_perm_equals_b_evidence_" in source
    assert "fnnls_kernel_injected" in source
    assert '_fnnls_injection_counts["calls"] <= 0' in source


def test_the_nnls_warm_start_flag_gates_only_the_within_block_clears(cell_ns):
    """The between-row clears are unconditional; only the in-block ones are gated.

    Dense and sparse-numba S3 key IDENTICALLY, so a memo carried between rows
    would let one formalism inherit the other's passive set — a channel
    production does not have. A memo carried between the calls of one block is
    exactly what a sampler's successive evaluations do, which is the thing the
    flag exists to measure.
    """
    source = CELL_PATH.read_text()
    tree = ast.parse(source)

    assert (
        '_cell_parser.add_argument("--nnls-warm-start", choices=("off", "on"), default="off")'
        in (source)
    ), "the flag must default to off"
    assert '_os.environ["AUTOARRAY_NNLS_WARM_START"] = "1" if NNLS_WARM_START_ON else "0"' in source

    # `_clear_memos` itself is never gated...
    clear = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_clear_memos"
    )
    assert "NNLS_WARM_START" not in ast.unparse(clear)

    # ...the block-level wrapper is.
    within = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_clear_memos_within_block"
    )
    within_src = ast.unparse(within)
    assert "NNLS_WARM_START_ON" in within_src and "_clear_memos()" in within_src

    # And the ABBA block calls the gated one, three times, while the row loop
    # keeps the unconditional one.
    abba = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_abba_blocks"
    )
    abba_src = ast.unparse(abba)
    assert abba_src.count("_clear_memos_within_block()") == 3
    assert "_clear_memos()" not in abba_src.replace("_clear_memos_within_block()", "")

    assert '"memo_cleared_within_block": not NNLS_WARM_START_ON' in source
    assert '"nnls_warm_start_mode": NNLS_WARM_START' in source


def test_analysis_imaging_is_built_with_use_jax_false():
    """A numba row built with ``use_jax=True`` would be a JAX row wearing the label."""
    tree = ast.parse(CELL_PATH.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AnalysisImaging"
    ]
    assert calls, "the cell never builds an AnalysisImaging"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "use_jax" in kwargs, "AnalysisImaging built without an explicit use_jax"
        assert isinstance(kwargs["use_jax"], ast.Constant)
        assert kwargs["use_jax"].value is False


def test_resolve_output_paths_is_called_with_an_explicit_cell():
    """Without ``cell=`` the first-token rule derives ``fixed`` and clobbers phase 0.

    ``_profile_cli.py:558-562`` documents this exact bug happening before
    (autolens_profiling#219, ``delaunay_nn`` deriving to ``delaunay``).
    """
    tree = ast.parse(CELL_PATH.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_output_paths"
    ]
    assert len(calls) == 1, f"expected exactly one resolve_output_paths call, got {len(calls)}"
    kwargs = {kw.arg for kw in calls[0].keywords}
    assert "cell" in kwargs, "resolve_output_paths called without an explicit cell="


def test_the_unattributed_row_is_explicit(cell_ns):
    """The remainder is a row of its own and is never folded into a neighbour."""
    source = CELL_PATH.read_text()
    assert cell_ns["UNATTRIBUTED_LABEL"] in source
    assert "_steps[UNATTRIBUTED_LABEL]" in source
    assert cell_ns["MAX_UNATTRIBUTED_FRACTION"] == 0.05
    assert cell_ns["MAX_INSTRUMENTATION_OVERHEAD_MS"] == 12.0
    assert cell_ns["REFERENCE_OVERHEAD_RATIO"] == 1.03


# ---------------------------------------------------------------------------
# 3. The measurement protocol: counterbalanced overhead, steady-state warm-up
# ---------------------------------------------------------------------------


def test_the_overhead_ratio_comes_from_counterbalanced_blocks(cell_ns):
    """The ratio is the mean of ABBA block ratios, never instrumented/clean means.

    A sequential estimator charges any drift over the row to the instrumentation.
    Measured on HST / Delaunay / N=484 it returned 0.71, 0.98, 1.10, 1.11, 1.24
    and 1.37 across six rows of one leg — including ratios below 1 — against a
    1.03 threshold. The counterbalanced estimator returned 1.0085 on the same
    configuration.
    """
    source = CELL_PATH.read_text()

    assert "_abba_blocks" in source, "the cell no longer runs a counterbalanced design"
    assert "_overhead_ratio = float(np.mean(_block_ratios))" in source, (
        "the overhead ratio must be the mean of the per-block ABBA ratios"
    )
    assert "_inst_mean / max(_clean_mean" not in source, (
        "the sequential instrumented/clean estimator is back; it cannot resolve a 3 % "
        "effect against this host's noise floor."
    )

    # The block really is A B B A: two clean calls straddling two instrumented.
    tree = ast.parse(source)
    abba = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_abba_blocks"
    )
    body = ast.unparse(abba)
    assert body.index("a1, value = _one_call") < body.index("b1, value_b1 = _one_call")
    assert body.index("b2, value_b2 = _one_call") < body.index("a2, value = _one_call")
    # `ast.unparse` normalises the parentheses, so match its own rendering.
    assert "(b1 + b2) / 2.0 / max((a1 + a2) / 2.0" in body


def test_the_overhead_gate_is_repeat_conditional(cell_ns):
    """Below the block threshold the ratio is RECORDED with its spread, not asserted.

    Asserting a 3 % threshold off one block is asserting noise. Recording it —
    with the observed clean-call spread beside it — keeps the comparison in the
    artifact without turning the host's scatter into a verdict.
    """
    source = CELL_PATH.read_text()

    assert cell_ns["MIN_BLOCKS_FOR_OVERHEAD_ASSERT"] == 3
    assert '_overhead_status == "FAIL"' in source, (
        "the assert must fire on the recorded status, not on a bare ratio comparison"
    )
    assert '"RECORDED"' in source
    assert "instrumentation_overhead_status" in source
    assert "clean_relative_spread" in source, (
        "the observed spread must be recorded beside the ratio it has to beat"
    )
    # The block threshold is untouched by phase 4 — the UNITS of the gate changed
    # (ratio -> milliseconds), not how many blocks it takes to render a verdict.
    assert cell_ns["MAX_INSTRUMENTATION_OVERHEAD_MS"] == 12.0
    assert cell_ns["REFERENCE_OVERHEAD_RATIO"] == 1.03


def _overhead_gate(cell_ns, *, ratio, clean_call_ms, n_blocks):
    """Run the cell's OWN overhead-gate statements, lifted from its AST.

    ``_overhead_ms`` and ``_overhead_status`` are assigned inside the row loop,
    which is module-level code this test cannot import. The two assignments are
    therefore lifted by name and executed against a prepared namespace, so what
    is exercised below is the cell's arithmetic and the cell's branch, not a
    re-implementation of them that could drift from it.
    """
    tree = ast.parse(CELL_PATH.read_text())
    wanted = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ("_overhead_ms", "_overhead_status")
            for t in node.targets
        )
    ]
    assert len(wanted) == 2, (
        f"expected exactly one _overhead_ms and one _overhead_status assignment, "
        f"found {len(wanted)}"
    )
    wanted.sort(key=lambda node: node.lineno)

    namespace = {
        "_overhead_ratio": ratio,
        "_clean_mean": clean_call_ms / 1e3,
        "_abba": {"n_blocks": n_blocks},
        "MIN_BLOCKS_FOR_OVERHEAD_ASSERT": cell_ns["MIN_BLOCKS_FOR_OVERHEAD_ASSERT"],
        "MAX_INSTRUMENTATION_OVERHEAD_MS": cell_ns["MAX_INSTRUMENTATION_OVERHEAD_MS"],
        "_overhead_assertable": n_blocks >= cell_ns["MIN_BLOCKS_FOR_OVERHEAD_ASSERT"],
    }
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(CELL_PATH), "exec"),
        namespace,
    )
    return namespace["_overhead_ms"], namespace["_overhead_status"]


def test_the_overhead_gate_is_a_millisecond_budget(cell_ns):
    """The same ratio passes on a short call and fails on a long one.

    ``call_accounting``'s cost is a fixed number of wrapper invocations per call,
    so the ratio it produces is a function of the call LENGTH as much as of the
    instrument. Read as a ratio the gate therefore tightened every time this
    campaign made the call shorter, and on RAL it killed the shorter of two rows
    for costing *more* instrument in absolute terms than the longer one that
    passed: x1.0147 at 413 ms is 6.1 ms and cleared 1.03; x1.0366 at 224 ms is
    8.2 ms and did not.

    The three rows below are those measurements. The middle one is the
    counterfactual: the SAME 1.037 ratio on a 400 ms call really is 14.8 ms of
    instrument, and must still fail.
    """
    budget = cell_ns["MAX_INSTRUMENTATION_OVERHEAD_MS"]
    assert budget == 12.0

    # Job 343355, feature arm: the row the ratio gate killed.
    overhead_ms, status = _overhead_gate(cell_ns, ratio=1.037, clean_call_ms=224.0, n_blocks=32)
    assert overhead_ms == pytest.approx(8.288, rel=1e-6)
    assert overhead_ms <= budget
    assert status == "PASS", (
        f"1.037 at a 224 ms call is {overhead_ms:.2f} ms of instrument — inside the "
        f"{budget} ms budget, and the row the old ratio gate destroyed"
    )

    # The counterfactual: the same ratio on the call the 1.03 was calibrated on.
    overhead_ms, status = _overhead_gate(cell_ns, ratio=1.037, clean_call_ms=400.0, n_blocks=32)
    assert overhead_ms == pytest.approx(14.8, rel=1e-6)
    assert status == "FAIL", (
        f"1.037 at a 400 ms call is {overhead_ms:.2f} ms — over the {budget} ms budget. "
        f"A budget that passed this would not be a gate."
    )

    # Job 343356, the 413 ms row that passed the ratio gate. It must still pass.
    overhead_ms, status = _overhead_gate(cell_ns, ratio=1.0147, clean_call_ms=413.0, n_blocks=32)
    assert overhead_ms == pytest.approx(6.0711, rel=1e-6)
    assert status == "PASS"

    # And the block threshold still overrides the verdict, in milliseconds too.
    overhead_ms, status = _overhead_gate(cell_ns, ratio=1.037, clean_call_ms=400.0, n_blocks=1)
    assert overhead_ms == pytest.approx(14.8, rel=1e-6)
    assert status == "RECORDED", (
        "one block cannot resolve the budget any better than it could resolve the ratio"
    )


def test_the_overhead_gate_records_both_the_ms_and_the_ratio(cell_ns):
    """The ratio does not disappear — levers 1-3 are chained against it.

    The note reads 1.0147 at 413.301 ms, 1.0167 at 302.709, 1.0221 at 267.448 and
    so on. A JSON that published only the new milliseconds would make those rows
    unreadable against this one.
    """
    source = CELL_PATH.read_text()

    assert '"instrumentation_overhead_ratio": _overhead_ratio,' in source
    assert '"instrumentation_overhead_ms": _overhead_ms,' in source
    assert '"instrumentation_overhead_threshold_ms": MAX_INSTRUMENTATION_OVERHEAD_MS,' in source
    assert '"instrumentation_overhead_reference_ratio": REFERENCE_OVERHEAD_RATIO,' in source
    assert '"max_instrumentation_overhead_ms": MAX_INSTRUMENTATION_OVERHEAD_MS,' in source
    assert '"min_blocks_for_overhead_assert": MIN_BLOCKS_FOR_OVERHEAD_ASSERT,' in source

    # The gate compares milliseconds, not the ratio. A ratio comparison left in
    # the status expression would be the old gate wearing the new name.
    assert "_overhead_ms <= MAX_INSTRUMENTATION_OVERHEAD_MS" in source
    assert "_overhead_ratio <= " not in source, (
        "the status must be decided on the millisecond budget, not on the ratio"
    )
    assert "_overhead_ms = (_overhead_ratio - 1.0) * _clean_mean * 1e3" in source, (
        "the overhead milliseconds must come from the row's own measured clean mean"
    )


def test_the_warmup_runs_to_steady_state_and_records_every_call(cell_ns):
    """A fixed warm-up count assumes what it should measure; the sequence is evidence."""
    source = CELL_PATH.read_text()

    assert "_warm_to_steady_state" in source
    assert cell_ns["WARMUP_WINDOW"] == 3
    assert cell_ns["WARMUP_TOLERANCE"] == 0.10
    assert cell_ns["WARMUP_MAX_CALLS"] == 12
    assert cell_ns["WARMUP_MAX_CALLS"] >= 2 * cell_ns["WARMUP_WINDOW"], (
        "the cap must allow at least two windows or the steady-state test can never run"
    )

    # The whole sequence reaches the JSON, and the unsettled case is visible.
    assert '"sequence_s": sequence' in source
    assert '"steady": steady' in source
    assert '"warmup": _warmup' in source
    assert "NEVER SETTLED" in source, (
        "a warm-up that did not settle must say so in the log, not pass silently"
    )


def test_warm_to_steady_state_stops_on_a_flat_sequence_and_caps_on_a_ramp(cell_ns):
    """The loop's own logic, on synthetic sequences with known answers."""
    ns = _cell_namespace()

    # Rebind the two module-level helpers the loop calls, so the sequence is ours.
    flat = iter([0.30, 0.31, 0.29, 0.30, 0.31, 0.30, 0.30] + [0.30] * 20)
    ramp = iter([1.0 / (1.0 + 0.5 * i) for i in range(40)])

    for name, stream, expect_steady in (("flat", flat, True), ("ramp", ramp, False)):
        ns["_one_call"] = lambda _a, _r, _s=stream: (next(_s), 0.0)
        ns["_clear_memos"] = lambda: None
        ns["np"] = np
        ns["WARMUP_WINDOW"] = cell_ns["WARMUP_WINDOW"]
        ns["WARMUP_TOLERANCE"] = cell_ns["WARMUP_TOLERANCE"]
        ns["WARMUP_MAX_CALLS"] = cell_ns["WARMUP_MAX_CALLS"]

        tree = ast.parse(CELL_PATH.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_warm_to_steady_state"
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(CELL_PATH), "exec"), ns)

        out = ns["_warm_to_steady_state"](None, "b")
        assert out["steady"] is expect_steady, f"{name}: steady={out['steady']}"
        assert len(out["sequence_s"]) == out["n_calls"]
        assert out["first_call_incl_numba_compile_s"] == out["sequence_s"][0]
        if not expect_steady:
            assert out["n_calls"] == cell_ns["WARMUP_MAX_CALLS"], (
                "a sequence that never settles must run to the cap and say so"
            )


def test_s4b_resets_and_primes_the_same_timed_stream_before_abba():
    """Variable warm-up lengths cannot shift b and d_perm onto different iid draws."""
    source = CELL_PATH.read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_s4b_timed_stream"
    )
    events = []
    call_index = {"next": 99}

    class FakeAnalysis:
        def log_likelihood_function(self, *, instance):
            events.append(("prime", instance, call_index["next"]))
            return 0.0

    namespace = {
        "_clear_memos": lambda: events.append(("clear",)),
        "_call_index": call_index,
        "_instance_for": lambda route, index: f"{route}:{index}",
        "_IID_SEED": 263,
        "INSTANCE_MODE": "iid",
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(CELL_PATH), "exec"),
        namespace,
    )
    metadata = namespace["_prepare_s4b_timed_stream"](FakeAnalysis(), "b", 128)
    events.append(("abba", call_index["next"]))

    assert events == [("clear",), ("prime", "b:0", 0), ("abba", 1)]
    assert metadata["memo_reset_after_warmup"] is True
    assert metadata["memo_primed"] is True
    assert metadata["priming_instance_index"] == 0
    assert metadata["timed_instance_start_index"] == 1
    assert metadata["timed_instance_count"] == 128
    assert metadata["iid_seed"] == 263
    assert source.index("_timed_stream = _prepare_s4b_timed_stream(") < source.index(
        "_abba = _abba_blocks(_analysis, _route, _n_blocks, _site_spec)"
    )


def test_the_artifact_declares_its_own_timing_status_and_contention():
    """Numbers never meant to be quoted must say so in the artifact, not in a note.

    A smoke leg's milliseconds are wiring evidence. A reader who skips the prose
    and reads the JSON must still be told that — and must be able to see whether
    the host was busy while the rows were timed, because a load average above the
    core count means the calls were queueing for the cores they were measured on.
    """
    source = CELL_PATH.read_text()

    assert '"timing_status": _timing_status' in source
    assert '"wiring_only_not_measured"' in source
    assert '"measured_under_contention"' in source
    assert '"contention_warning": _contended' in source
    assert "load_average_at_start" in source and "load_average_at_end" in source
    assert "/proc/loadavg" in source

    # The smoke declaration is derived from the run's own config name, not from a
    # flag someone has to remember to pass.
    assert '"smoke" in _cli.config_name.lower()' in source


def test_the_ramp_hypothesis_is_recorded_as_falsified():
    """The 15-call sequence and its verdict live in the cell, not only in a report.

    An unexplained 2.9x transient left lying around is an invitation for the next
    person to re-derive it as a warm-up ramp and add warm-up calls that buy
    nothing. The measurement that falsified it is recorded beside the loop it
    would otherwise justify.
    """
    source = CELL_PATH.read_text()
    # Whitespace-normalised: a docstring reflow must not be able to fail this.
    flat = " ".join(source.split())

    assert "THERE IS NO WARM-UP RAMP ON THIS CONFIGURATION" in flat
    for value in (
        "341, 310, 341, 376, 350, 337, 364, 344, 294, 279, 319, 338, 374, 367, 362 ms",
        "first/last 0.942",
        "It was queueing, not warming.",
    ):
        assert value in flat, f"the ramp evidence no longer records {value!r}"


# ---------------------------------------------------------------------------
# 4. Route d_np: the injection seam, over a real fit
# ---------------------------------------------------------------------------


def test_numpy_solver_injected_fires_on_the_numpy_path_and_delegates_jax():
    """The counters are the evidence the patch took, and the JAX branch is inverted.

    ``certified_solver_injected`` dispatches on JAX and delegates numpy; this one
    dispatches on numpy and delegates JAX. A numba cell only ever produces the
    first kind of call, which is exactly why the second is tested here — nothing
    in that process could ever notice it being mis-routed.

    The library function is replaced by a spy for the duration, so the delegated
    call is *observed* rather than executed: the real JAX branch would import a
    JAX runtime, which is the one thing this whole module exists to avoid.
    """
    from autoarray.inversion.inversion import inversion_util

    library = inversion_util.reconstruction_positive_only_from
    spy_calls = {"n": 0, "xp": None}

    def spy(data_vector, curvature_reg_matrix, settings=None, xp=np, fingerprint=None):
        spy_calls["n"] += 1
        spy_calls["xp"] = xp
        return "delegated-untouched"

    seen = {}

    def fake_solver(crm, dv, *, stats=None):
        seen["crm_shape"] = np.asarray(crm).shape
        if stats is not None:
            stats["n_factorisations"] = 1
        return np.zeros(np.asarray(dv).shape[0])

    crm = np.eye(3) * 2.0
    dv = np.array([1.0, 2.0, 3.0])

    class _FakeXp:
        pass

    fake_jax = _FakeXp()
    fake_jax.__name__ = "jax.numpy"

    inversion_util.reconstruction_positive_only_from = spy
    try:
        with flns.numpy_solver_injected(fake_solver, label="unit") as counts:
            patched = inversion_util.reconstruction_positive_only_from
            assert patched is not spy
            assert patched.__wrapped__ is spy

            out = patched(data_vector=dv, curvature_reg_matrix=crm, settings=None, xp=np)
            assert counts["numpy"] == 1 and counts["jax"] == 0
            assert np.array_equal(out, np.zeros(3))
            assert seen["crm_shape"] == (3, 3)
            assert counts["last_stats"]["n_factorisations"] == 1
            assert spy_calls["n"] == 0, "a numpy call must not reach the library"

            delegated = patched(
                data_vector=dv, curvature_reg_matrix=crm, settings=None, xp=fake_jax
            )
            assert counts["jax"] == 1 and counts["numpy"] == 1
            assert delegated == "delegated-untouched"
            assert spy_calls["n"] == 1 and spy_calls["xp"] is fake_jax

        assert inversion_util.reconstruction_positive_only_from is spy
    finally:
        inversion_util.reconstruction_positive_only_from = library


def test_numpy_solver_injected_restores_by_identity_and_refuses_a_stranger():
    """Restoring over someone else's rebinding would hide a nesting bug."""
    from autoarray.inversion.inversion import inversion_util

    original = inversion_util.reconstruction_positive_only_from
    stranger = object()

    with pytest.raises(RuntimeError, match="rebound by something else"):
        with flns.numpy_solver_injected(lambda *a, **k: None, label="unit"):
            inversion_util.reconstruction_positive_only_from = stranger

    assert inversion_util.reconstruction_positive_only_from is stranger
    inversion_util.reconstruction_positive_only_from = original


def test_route_d_np_reaches_route_bs_evidence_on_a_real_fit(tiny_s3_pair):
    """P4, on the fixture: the injected kernel is the library's answer, not near it.

    The whole ``FitImaging`` is run each way — the library's own mapper, ``F +
    lambda H``, both log determinants and evidence — with one function swapped.
    If this drifts, route ``d_np``'s milliseconds are for a different minimiser.
    """
    autolens = pytest.importorskip("autolens")
    al = autolens

    _dense, sparse = tiny_s3_pair

    def one_fit():
        _clear_nnls_memo()
        return al.FitImaging(
            dataset=sparse.dataset,
            tracer=sparse.source_only_tracer,
            adapt_images=sparse.fit.adapt_images,
            settings=sparse.fit.settings,
            xp=np,
        )

    fit_b = one_fit()
    fom_b = float(fit_b.figure_of_merit)
    x_b = np.asarray(fit_b.inversion.reconstruction, dtype=float)

    with flns.numpy_solver_injected(flns.nnls_factor_reuse, label="d_np") as counts:
        fit_d = one_fit()
        fom_d = float(fit_d.figure_of_merit)
        x_d = np.asarray(fit_d.inversion.reconstruction, dtype=float)

    assert counts["numpy"] > 0, "the injected solver never fired; this compared b with b"
    assert counts["jax"] == 0

    rel = abs(fom_d - fom_b) / max(abs(fom_b), 1e-300)
    assert rel <= P4_RTOL, (
        f"d_np figure_of_merit {fom_d!r} vs b {fom_b!r} (rel {rel:.3e} > {P4_RTOL:g})"
    )
    assert np.max(np.abs(x_d - x_b)) <= P4_RTOL * max(float(np.max(np.abs(x_b))), 1e-300)
