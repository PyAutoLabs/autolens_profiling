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
from likelihood_breakdown import fixed_light_system as fls  # noqa: E402

CELL_PATH = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_numba.py"

P2_RTOL = 1.0e-9
P3_RTOL = 1.0e-6


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
    "ALL_FORMALISM_KEYS",
    "GROUP_LABELS",
    "MAX_INSTRUMENTATION_OVERHEAD",
    "MIN_BLOCKS_FOR_OVERHEAD_ASSERT",
    "MAX_UNATTRIBUTED_FRACTION",
    "WARMUP_WINDOW",
    "WARMUP_TOLERANCE",
    "WARMUP_MAX_CALLS",
    "UNATTRIBUTED_LABEL",
    "P2_RTOL",
    "P3_RTOL",
    "MAPPER_LOGDET_RTOL",
)


def _cell_namespace() -> dict:
    """Execute the cell's helpers + constants, and nothing else, in a fresh dict.

    The names the lifted ``_site_spec`` closes over are the library classes the
    cell imports; they are supplied here so the spec under test is the *real*
    one. A site the cell adds without this test seeing it is a site whose
    ``cached`` declaration is never checked.
    """
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

    namespace = {
        "call_accounting": ca,
        "inversion_util": inversion_util,
        "fnnls_module": fnnls_module,
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
    threshold = cell_ns["MAX_INSTRUMENTATION_OVERHEAD"]
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


def test_curvature_reg_matrix_is_recomputed_more_than_once(tiny_s3_pair, cell_ns):
    """``F + lambda H`` is a plain ``property`` and the library rebuilds it repeatedly.

    Pinned deliberately. If a future PyAutoArray change caches it, this fails and
    the ``F + lambda H`` row of the decomposition is updated on purpose rather
    than quietly becoming a different quantity.
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

    assert snapshot["inversion.curvature_reg_matrix"]["n_calls"] >= 2, (
        "AbstractInversion.curvature_reg_matrix was reached "
        f"{snapshot['inversion.curvature_reg_matrix']['n_calls']} time(s). If PyAutoArray "
        "now caches it, update the decomposition's F + lambda H row on purpose."
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


def test_all_route_keys_is_exactly_a_b_c(cell_ns):
    """No d/d0/e: those are the JAX certified-active-set rows of another cell."""
    assert cell_ns["ALL_ROUTE_KEYS"] == ("a", "b", "c")
    assert cell_ns["_parse_routes"](None) == ("a", "b", "c")
    assert cell_ns["_parse_routes"]("c,a") == ("a", "c")
    with pytest.raises(ValueError, match="unknown route"):
        cell_ns["_parse_routes"]("a,d")
    assert cell_ns["ALL_FORMALISM_KEYS"] == ("dense", "sparse_numba")
    assert cell_ns["_parse_formalisms"]("both") == ("dense", "sparse_numba")
    assert cell_ns["_parse_formalisms"]("dense") == ("dense",)


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
    assert cell_ns["MAX_INSTRUMENTATION_OVERHEAD"] == 1.03


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
    # And the threshold itself is untouched — the estimator was the fix.
    assert cell_ns["MAX_INSTRUMENTATION_OVERHEAD"] == 1.03


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
