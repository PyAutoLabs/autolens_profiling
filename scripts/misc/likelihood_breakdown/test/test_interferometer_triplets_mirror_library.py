"""Interferometer / datacube W~ triplets mirror the library (autolens_profiling #458 follow-up).

#458 made ``sparse_steps.sparse_triplets`` (imaging) pass ``pix_sizes_for_sub``, because
``Mapper.sparse_triplets_data`` / ``_curvature`` pass it since PyAutoArray #643. Three
interferometer-side scripts build their own triplets too:

- ``scripts/misc/likelihood_breakdown/interferometer_pixelized.py`` (``sparse_triplets``)
- ``scripts/datacube/delaunay/likelihood_breakdown.py`` (sparse path)
- ``scripts/interferometer/delaunay/fft_size_levers.py`` (``triplets``)

All three build the extent-grid curvature triplets (``return_rows_slim=False``) that feed
``InterferometerSparseOperator.curvature_matrix_diag_from``. The library's counterpart is
``InversionInterferometerSparse._sparse_triplets_curvature_from``, which (unlike the slim
data-vector triplets) does **not** pass ``pix_sizes_for_sub``: it relies on the operator
dropping the Delaunay ``-1`` padding, which carries weight 0. So the scripts are already
exact mirrors and need no ``pix_sizes_for_sub``. These tests pin that:

1. each script's ``sparse_triplets_from`` call has exactly the library call's keywords
   (less ``slim_weights``, the primary beam, which none of these datasets has), read from the
   installed library source, so drift on either side fails here;
2. every ``curvature_matrix_diag_from`` call in them passes ``xp=jnp`` (the JAX branch);
3. numerically, on a real over-sampled Delaunay mapper whose stencils carry ``-1`` padding,
   the script's unmasked triplets give the same ``F`` through the JAX operator as the masked
   ones, and both equal the dense mapping-formalism inversion's curvature matrix.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import autoarray as aa  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from autoarray.inversion.inversion.interferometer.sparse import (  # noqa: E402
    InversionInterferometerSparse,
)
from autoarray.inversion.mappers import mapper_util  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

SCRIPTS = (
    "scripts/misc/likelihood_breakdown/interferometer_pixelized.py",
    "scripts/datacube/delaunay/likelihood_breakdown.py",
    "scripts/interferometer/delaunay/fft_size_levers.py",
)


def _calls(tree, name):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "attr", None) == name or getattr(n.func, "id", None) == name)
    ]


def _library_curvature_triplet_kwargs() -> dict[str, str]:
    source = textwrap.dedent(
        inspect.getsource(InversionInterferometerSparse._sparse_triplets_curvature_from)
    )
    (call,) = _calls(ast.parse(source), "sparse_triplets_from")
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_the_library_curvature_triplets_take_no_pix_sizes():
    """If this fails the library changed: make the three scripts pass what it now passes."""
    kwargs = _library_curvature_triplet_kwargs()
    assert "pix_sizes_for_sub" not in kwargs
    assert kwargs["return_rows_slim"] == "False"
    assert kwargs["xp"] == "self._xp"


@pytest.mark.parametrize("script", SCRIPTS)
def test_each_script_builds_the_library_curvature_triplets(script):
    library = set(_library_curvature_triplet_kwargs()) - {"slim_weights"}
    calls = _calls(ast.parse((ROOT / script).read_text()), "sparse_triplets_from")
    assert len(calls) == 1, script
    kwargs = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert set(kwargs) == library, script
    assert kwargs["return_rows_slim"] == "False", script
    assert kwargs["xp"] == "jnp", script


@pytest.mark.parametrize("script", SCRIPTS)
def test_each_script_runs_the_operator_on_its_jax_branch(script):
    calls = _calls(ast.parse((ROOT / script).read_text()), "curvature_matrix_diag_from")
    assert calls, script
    for call in calls:
        kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kwargs.get("xp") == "jnp", f"{script}:{call.lineno}"


def _setup():
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0)
    rng = np.random.default_rng(seed=11)
    n_vis = 64
    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=rng.normal(size=(n_vis, 2))),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((n_vis, 2))),
        # ~0.5 / delta_rad: phases span [-pi, pi], so W~ is not near-constant
        uv_wavelengths=1.0e5 * rng.uniform(-1.0, 1.0, size=(n_vis, 2)),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )
    operator = dataset.apply_sparse_operator(
        nufft_precision_operator=dataset.psf_precision_operator_from(method="numpy"),
        batch_size=4,
    ).sparse_operator

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)
    mesh_grid = aa.image_mesh.Overlay(shape=(4, 4)).image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )
    interpolator = aa.mesh.Delaunay(pixels=mesh_grid.shape[0]).interpolator_from(
        source_plane_data_grid=grid, source_plane_mesh_grid=mesh_grid
    )
    # regularized, so the dense inversion adds no `no_regularization` 1e-3 to F's diagonal
    mapper = aa.Mapper(interpolator=interpolator, regularization=aa.reg.Constant(coefficient=1.0))
    return mask, dataset, operator, mapper


def _triplets(mask, mapper, **extra):
    # the scripts' call, verbatim
    return mapper_util.sparse_triplets_from(
        pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
        pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
        fft_index_for_masked_pixel=mask.extent_index_for_masked_pixel,
        sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
        return_rows_slim=False,
        xp=jnp,
        **extra,
    )


def test_unmasked_curvature_triplets_give_the_dense_curvature_matrix():
    mask, dataset, operator, mapper = _setup()
    S = mapper.params

    rows, cols, vals = _triplets(mask, mapper)
    # the fixture holds what it claims: Delaunay -1 padding, at weight 0, survives unmasked
    padded = np.asarray(cols) < 0
    assert padded.any()
    assert (np.asarray(vals)[padded] == 0.0).all()
    assert S > operator.batch_size

    F_script = operator.curvature_matrix_diag_from(rows=rows, cols=cols, vals=vals, S=S, xp=jnp)
    F_masked = operator.curvature_matrix_diag_from(
        *_triplets(mask, mapper, pix_sizes_for_sub=mapper.pix_sizes_for_sub_slim_index),
        S=S,
        xp=jnp,
    )
    F_dense = aa.Inversion(dataset=dataset, linear_obj_list=[mapper]).curvature_matrix

    np.testing.assert_array_equal(np.asarray(F_script), np.asarray(F_masked))
    np.testing.assert_allclose(
        np.asarray(F_script), np.asarray(F_dense), rtol=1e-8, atol=1e-10 * np.abs(F_dense).max()
    )
