"""Sparse triplets and the regression-pin gate (likelihood-lifecycle 2a, autolens_profiling#456).

1. ``sparse_steps.sparse_triplets`` must mask invalid stencil slots exactly as
   ``Mapper.sparse_triplets_data`` / ``sparse_triplets_curvature`` do since
   PyAutoArray #643 (``pix_sizes_for_sub``): a Delaunay mesh writes ``-1`` into
   unused stencil slots, and unmasked those reach the data-vector scatter-add
   (``np.bincount`` refuses them — the batch-0 crash — and a JAX ``.at[-1]``
   would silently wrap onto the last source pixel). The D and mapped-image
   steps must also dispatch to the JAX branch (``xp=jnp``) so they jit.
   Checked against the dense ``mapping_matrix_from`` reference on a tiny
   hand-built stencil.

2. A breakdown's regression pin is asserted only for the instrument it was
   measured on (hst). The MGE breakdown asserted the hst pin for every
   ``--instrument`` (ao failed: 659642 vs 27373). Scripts are read as AST, not run.
"""

from __future__ import annotations

import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from autoarray.inversion.mappers import mapper_util  # noqa: E402
from likelihood_breakdown import sparse_steps  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

#: 4 slim pixels, sub-size 1, a 3-slot stencil over S=3 source pixels. Pixel 1
#: uses 2 slots and pads the third with -1 (Delaunay-style); pixel 3 uses 1 slot
#: and leaves stale indices (not -1) in the unused ones.
PIX_INDEXES = np.array([[0, 1, 2], [2, 0, -1], [1, 2, 0], [2, 1, 1]], dtype=np.int32)
PIX_WEIGHTS = np.array(
    [[0.2, 0.3, 0.5], [0.6, 0.4, 0.9], [0.1, 0.1, 0.8], [1.0, 0.7, 0.7]], dtype=np.float64
)
PIX_SIZES = np.array([3, 2, 3, 1], dtype=np.int32)
SLIM_INDEX_FOR_SUB = np.arange(4, dtype=np.int32)
SUB_FRACTION = np.ones(4, dtype=np.float64)
FFT_INDEX = np.array([0, 1, 2, 3], dtype=np.int32)
S = 3


def _triplets():
    return sparse_steps.sparse_triplets(
        jnp.asarray(PIX_INDEXES),
        jnp.asarray(PIX_WEIGHTS),
        jnp.asarray(SLIM_INDEX_FOR_SUB),
        jnp.asarray(FFT_INDEX),
        jnp.asarray(SUB_FRACTION),
        jnp.asarray(PIX_SIZES),
    )


def _dense_mapping_matrix():
    return mapper_util.mapping_matrix_from(
        pix_indexes_for_sub_slim_index=PIX_INDEXES,
        pix_size_for_sub_slim_index=PIX_SIZES,
        pix_weights_for_sub_slim_index=PIX_WEIGHTS,
        pixels=S,
        total_mask_pixels=4,
        slim_index_for_sub_slim_index=SLIM_INDEX_FOR_SUB,
        sub_fraction=SUB_FRACTION,
    )


def test_triplets_mask_padding_and_unused_slots_to_column_zero_value_zero():
    rows_data, cols, vals, rows_curv = _triplets()
    cols, vals = np.asarray(cols), np.asarray(vals)
    assert (cols >= 0).all() and (cols < S).all()
    # the -1 slot of pixel 1 and the two unused slots of pixel 3 contribute nothing
    slot = np.arange(PIX_INDEXES.size) % PIX_INDEXES.shape[1]
    pixel = np.arange(PIX_INDEXES.size) // PIX_INDEXES.shape[1]
    dropped = slot >= PIX_SIZES[pixel]
    assert dropped.sum() == 3
    assert (vals[dropped] == 0.0).all() and (cols[dropped] == 0).all()
    # the scattered operator is the dense mapping matrix the library builds
    A = np.zeros((4, S))
    np.add.at(A, (np.asarray(rows_data), cols), vals)
    np.testing.assert_allclose(A, _dense_mapping_matrix(), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(rows_curv), FFT_INDEX[np.asarray(rows_data)])


def test_the_library_mapper_triplets_are_what_the_script_mirrors():
    """Same arguments as ``Mapper.sparse_triplets_data`` (abstract.py) -> same triplets."""
    lib = mapper_util.sparse_triplets_from(
        pix_indexes_for_sub=jnp.asarray(PIX_INDEXES),
        pix_weights_for_sub=jnp.asarray(PIX_WEIGHTS),
        slim_index_for_sub=jnp.asarray(SLIM_INDEX_FOR_SUB),
        fft_index_for_masked_pixel=jnp.asarray(FFT_INDEX),
        sub_fraction_slim=jnp.asarray(SUB_FRACTION),
        pix_sizes_for_sub=jnp.asarray(PIX_SIZES),
        xp=jnp,
    )
    for ours, theirs in zip(_triplets()[:3], lib):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


def _ctx():
    # data_vector reads only the parameter layout: mapper block [0, 3), one MGE column at 3.
    return SimpleNamespace(
        total_params=S + 1,
        mapper_start=0,
        mapper_end=S,
        func_start=S,
        func_end=S + 1,
        mapper_params=S,
    )


def test_data_vector_is_the_dense_reference_eager_and_under_jit():
    rows_data, cols, vals, _ = _triplets()
    psf_weighted = jnp.asarray([1.5, -2.0, 0.25, 3.0])
    operated = jnp.asarray([[1.0], [2.0], [0.5], [0.0]])
    image = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    noise = jnp.asarray([1.0, 2.0, 1.0, 2.0])
    args = (psf_weighted, rows_data, cols, vals, operated, image, noise)
    fn = partial(sparse_steps.data_vector, _ctx())
    expected = np.concatenate(
        [
            _dense_mapping_matrix().T @ np.asarray(psf_weighted),
            np.asarray(operated).T @ (np.asarray(image) / np.asarray(noise) ** 2),
        ]
    )
    np.testing.assert_allclose(np.asarray(fn(*args)), expected, rtol=1e-14)
    # jit: the numpy (xp=np) branch would hit np.asarray on a tracer
    np.testing.assert_allclose(np.asarray(jax.jit(fn)(*args)), expected, rtol=1e-14)


def test_unmasked_triplets_are_what_crashed_batch_zero():
    """Control: without pix_sizes_for_sub the -1 column survives and numpy bincount refuses it."""
    from autoarray.inversion.inversion.imaging import inversion_imaging_util

    rows, cols, vals = mapper_util.sparse_triplets_from(
        pix_indexes_for_sub=PIX_INDEXES,
        pix_weights_for_sub=PIX_WEIGHTS,
        slim_index_for_sub=SLIM_INDEX_FOR_SUB,
        fft_index_for_masked_pixel=FFT_INDEX,
        sub_fraction_slim=SUB_FRACTION,
    )
    assert (np.asarray(cols) < 0).any()
    with pytest.raises(ValueError, match="negative"):
        inversion_imaging_util.data_vector_via_psf_weighted_data_from(
            psf_weighted_data=np.ones(4), rows=rows, cols=cols, vals=vals, S=S
        )


def test_every_sparse_breakdown_passes_the_mapper_pix_sizes():
    for cell in ("delaunay", "delaunay_nn", "rectangular"):
        text = (ROOT / "scripts/imaging" / cell / "likelihood_breakdown.py").read_text()
        assert "inv_mapper.pix_sizes_for_sub_slim_index" in text, cell


# --------------------------------------------------------------- the pin gate

#: Breakdown script -> its HST-only pin constant.
HST_PINNED_BREAKDOWNS = {
    "scripts/imaging/delaunay/likelihood_breakdown.py": "EXPECTED_LOG_EVIDENCE_HST",
    "scripts/imaging/delaunay_nn/likelihood_breakdown.py": "EXPECTED_LOG_EVIDENCE_HST",
    "scripts/imaging/rectangular/likelihood_breakdown.py": "EXPECTED_LOG_EVIDENCE_HST",
}


def _uses(node, name):
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


@pytest.mark.parametrize("script,pin", HST_PINNED_BREAKDOWNS.items())
def test_an_hst_only_pin_is_compared_only_behind_an_instrument_gate(script, pin):
    tree = ast.parse((ROOT / script).read_text())
    gates = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and "instrument != 'hst'" in ast.unparse(n.test)
    ]
    assert len(gates) == 1, f"{script}: no `instrument != 'hst'` gate before the pin"
    gate = gates[0]
    assert not any(_uses(n, pin) for n in gate.body), "the skip branch must not read the pin"
    # every module-level statement reading the pin, other than its assignment, sits in the gate's else
    for node in tree.body:
        if node is gate or isinstance(node, ast.Assign) and _uses(node.targets[0], pin):
            continue
        assert not _uses(node, pin), f"{script}:{node.lineno} reads {pin} outside the gate"


def test_mge_breakdown_looks_its_pin_up_by_instrument():
    tree = ast.parse((ROOT / "scripts/imaging/mge/likelihood_breakdown.py").read_text())
    assigns = {
        t.id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "EXPECTED_LOG_LIKELIHOOD_HST" not in assigns
    pins = ast.literal_eval(assigns["EXPECTED_LOG_LIKELIHOOD"])
    assert set(pins) == {"hst"}
    assert ast.unparse(assigns["_pinned_expected"]) == "EXPECTED_LOG_LIKELIHOOD.get(instrument)"
    # the assertion only runs in the else of `if _pinned_expected is None:`
    gate = next(
        n
        for n in tree.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "_pinned_expected is None"
    )
    assert "assert_allclose" not in ast.unparse(gate.body[0])
    assert "assert_allclose" in "".join(ast.unparse(n) for n in gate.orelse)
    others = [
        n
        for n in tree.body
        if n is not gate and "assert_allclose(log_likelihood_ref" in ast.unparse(n)
    ]
    assert not others


def test_every_sparse_breakdown_times_the_operator_on_the_standard_size_path():
    """``--standard-size --sparse`` left ``sparse_operator_build_s`` None -> float(None) crash."""
    for cell in ("delaunay", "delaunay_nn", "rectangular"):
        tree = ast.parse((ROOT / "scripts/imaging" / cell / "likelihood_breakdown.py").read_text())
        gate = next(
            n
            for n in tree.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "_cli.standard_size"
        )
        body = ast.unparse(gate)
        call = next(
            n
            for n in ast.walk(gate)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "standard_jax_dataset"
        )
        assert ast.unparse(call.args[3]) == "None", cell
        assert "sparse_operator_build_s = timer.records[-1][1]" in body, cell


def test_rectangular_constant_assembly_diagnostic_only_runs_for_constant_regularization():
    """The standard-size rectangular cell is ``Adapt`` (no ``coefficient``): the
    constant-formula diagnostic must be skipped, not crash, and write null."""
    text = (ROOT / "scripts/imaging/rectangular/likelihood_breakdown.py").read_text()
    tree = ast.parse(text)
    assert "pixelization_obj.regularization.coefficient if _reg_is_constant else None" in text
    gate = next(
        n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "_reg_is_constant"
    )
    assert "compute_regularization_matrix" in ast.unparse(gate.body[0])
    assert "reg_matrix_assembly_per_call = None" in "".join(ast.unparse(n) for n in gate.orelse)
