"""Unit tests for the phase-3 PSF-cube candidates (``psf_cube_injection.py``, #295).

**CPU JAX in fp64, no imaging dataset, no GPU.** Every fp64 candidate must be the
library's answer — the NumPy real-space path
(``Convolver.convolved_mapping_matrix_via_real_space_np_from``, scipy
``convolve(mode="same")``) is the ground truth — to 1e-12 absolute on two
fixtures: PyAutoArray's own 6x6 mask / asymmetric 3x3 kernel / 3-column example
(``test_autoarray/operators/test_convolver.py``), and a 40x40 circular mask with
a 21x21 Gaussian (the HST PSF size) and 50 random columns.

What a wrong candidate would look like, and what catches it: a kernel flipped
the wrong way or rolled by the wrong offset moves every blurred pixel by one —
``test__every_fp64_candidate_is_the_library_answer`` on the ASYMMETRIC 3x3
kernel; an even kernel centred like ``lax``'s ``SAME`` rather than scipy's —
``test__the_batched_conv_centres_an_even_kernel_like_scipy``; a patch that is
never taken, or never removed — the counts / restore tests.
"""

from __future__ import annotations

import inspect
import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

aa = pytest.importorskip("autoarray")


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import psf_cube_injection as pci  # noqa: E402

FP64 = [name for name, c in pci.CANDIDATES.items() if c.fp64_exact]
DIAGNOSTIC = [name for name, c in pci.CANDIDATES.items() if not c.fp64_exact]


def _small_fixture():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, True, True, True, True, True],
            ]
        ),
        pixel_scales=1.0,
    )
    kernel = aa.Array2D.no_mask(
        values=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]], pixel_scales=mask.pixel_scales
    )
    mapping_matrix = np.zeros((16, 3))
    mapping_matrix[7, 1] = 1.0
    mapping_matrix[9, 0] = 1.0
    mapping_matrix[10, 2] = 1.0
    return mask, kernel, mapping_matrix


def _hst_like_fixture(kernel_size: int = 21):
    mask = aa.Mask2D.circular(shape_native=(40, 40), pixel_scales=0.1, radius=1.5)
    y, x = np.mgrid[:kernel_size, :kernel_size] - (kernel_size - 1) / 2.0
    # Asymmetric on purpose: a flipped kernel must not pass by symmetry.
    values = np.exp(-(((y - 0.7) / 2.3) ** 2 + ((x + 0.4) / 3.1) ** 2))
    values /= values.sum()
    kernel = aa.Array2D.no_mask(values=values, pixel_scales=mask.pixel_scales)
    rng = np.random.default_rng(295)
    n_pix = int((~np.asarray(mask)).sum())
    mapping_matrix = rng.random((n_pix, 50))
    return mask, kernel, mapping_matrix


FIXTURES = {"pyautoarray_6x6_3x3": _small_fixture, "circular_40x40_21x21": _hst_like_fixture}


def _reference(kernel, mask, mapping_matrix):
    return np.asarray(
        aa.Convolver(kernel=kernel, use_fft=True).convolved_mapping_matrix_via_real_space_np_from(
            mapping_matrix=mapping_matrix, mask=mask
        )
    )


def _candidate(name, kernel, mask, mapping_matrix):
    convolver = aa.Convolver(kernel=kernel, use_fft=True)
    with pci.psf_convolution_injected(name) as counts:
        out = convolver.convolved_mapping_matrix_from(
            mapping_matrix=jnp.asarray(mapping_matrix), mask=mask, xp=jnp
        )
        out = np.asarray(out)
    return out, counts


def test__the_registry_names_every_candidate_and_marks_the_diagnostics():
    assert list(pci.CANDIDATES) == [
        "control",
        "frame_pow2",
        "layout_src_first",
        "real_space_direct",
        "conv_cudnn_batched",
        "mp_cube_c64",
        "c64_full",
    ]
    assert DIAGNOSTIC == ["mp_cube_c64", "c64_full"]
    for name, spec in pci.CANDIDATES.items():
        assert spec.name == name
        assert spec.kind in ("control", "lever", "diagnostic")
        assert (spec.kind == "diagnostic") == (not spec.fp64_exact)
        assert spec.library_change_if_wins


def test__the_cell_offers_exactly_the_registry():
    """argparse choices are a literal (the contracts test execs them standalone)."""
    import ast

    tree = ast.parse(
        (ROOT / "scripts/imaging/likelihood_breakdown/fixed_light_trace.py").read_text()
    )
    choices = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--psf-candidate"
        ):
            choices = next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "choices")
    assert choices is not None, "fixed_light_trace.py does not declare --psf-candidate"
    assert list(choices) == list(pci.CANDIDATES), (
        "fixed_light_trace.py's --psf-candidate choices no longer match CANDIDATES"
    )


@pytest.mark.parametrize("fixture", FIXTURES, ids=list(FIXTURES))
@pytest.mark.parametrize("name", FP64)
def test__every_fp64_candidate_is_the_library_answer(name, fixture):
    mask, kernel, mapping_matrix = FIXTURES[fixture]()
    reference = _reference(kernel, mask, mapping_matrix)
    got, counts = _candidate(name, kernel, mask, mapping_matrix)
    assert got.shape == reference.shape
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, reference, rtol=0.0, atol=1e-12)
    if name == "control":
        assert counts == {"jax": 0, "delegated": 0}
    else:
        assert counts == {"jax": 1, "delegated": 0}


@pytest.mark.parametrize("fixture", FIXTURES, ids=list(FIXTURES))
@pytest.mark.parametrize("name", DIAGNOSTIC)
def test__diagnostics_are_close_but_are_flagged_not_exact(name, fixture):
    mask, kernel, mapping_matrix = FIXTURES[fixture]()
    reference = _reference(kernel, mask, mapping_matrix)
    got, counts = _candidate(name, kernel, mask, mapping_matrix)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, reference, rtol=0.0, atol=1e-4)
    assert counts["jax"] == 1
    assert pci.CANDIDATES[name].fp64_exact is False
    assert pci.CANDIDATES[name].kind == "diagnostic"


@pytest.mark.parametrize("kernel_size", (4, 6))
def test__the_batched_conv_centres_an_even_kernel_like_scipy(kernel_size):
    mask, kernel, mapping_matrix = _hst_like_fixture(kernel_size)
    reference = _reference(kernel, mask, mapping_matrix)
    got, _ = _candidate("conv_cudnn_batched", kernel, mask, mapping_matrix)
    np.testing.assert_allclose(got, reference, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("name", [n for n in pci.CANDIDATES if n != "control"])
def test__numpy_xp_is_delegated_to_the_library(name):
    mask, kernel, mapping_matrix = _small_fixture()
    convolver = aa.Convolver(kernel=kernel, use_fft=True)
    reference = _reference(kernel, mask, mapping_matrix)
    with pci.psf_convolution_injected(name) as counts:
        got = convolver.convolved_mapping_matrix_from(mapping_matrix=mapping_matrix, mask=mask)
    assert counts == {"jax": 0, "delegated": 1}
    np.testing.assert_allclose(np.asarray(got), reference, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("name", [n for n in pci.CANDIDATES if n != "control"])
def test__a_blurring_matrix_is_delegated_to_the_library(name):
    mask, kernel, mapping_matrix = _small_fixture()
    convolver = aa.Convolver(kernel=kernel, use_fft=True)
    state = convolver.state_from(mask=mask)
    n_blur = int((~np.asarray(state.blurring_mask)).sum())
    blurring = np.full((n_blur, 3), 0.25)
    expected = convolver.convolved_mapping_matrix_from(
        mapping_matrix=jnp.asarray(mapping_matrix),
        mask=mask,
        blurring_mapping_matrix=jnp.asarray(blurring),
        xp=jnp,
    )
    with pci.psf_convolution_injected(name) as counts:
        got = convolver.convolved_mapping_matrix_from(
            mapping_matrix=jnp.asarray(mapping_matrix),
            mask=mask,
            blurring_mapping_matrix=jnp.asarray(blurring),
            xp=jnp,
        )
    assert counts == {"jax": 0, "delegated": 1}
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("name", [n for n, c in pci.CANDIDATES.items() if c.kind == "lever"])
def test__an_fp64_lever_asked_for_mixed_precision_is_delegated(name):
    mask, kernel, mapping_matrix = _small_fixture()
    convolver = aa.Convolver(kernel=kernel, use_fft=True)
    with pci.psf_convolution_injected(name) as counts:
        convolver.convolved_mapping_matrix_from(
            mapping_matrix=jnp.asarray(mapping_matrix),
            mask=mask,
            use_mixed_precision=True,
            xp=jnp,
        )
    assert counts == {"jax": 0, "delegated": 1}


def test__the_wrapper_covers_the_library_signature():
    from autoarray.operators.convolver import Convolver

    with pci.psf_convolution_injected("frame_pow2"):
        patched = Convolver.convolved_mapping_matrix_from
    library = set(inspect.signature(Convolver.convolved_mapping_matrix_from).parameters)
    assert library <= set(inspect.signature(patched).parameters)


def test__a_library_parameter_the_wrapper_lacks_fails_at_injection_time():
    def library(self, mapping_matrix, mask, new_knob=None):  # pragma: no cover
        return None

    def wrapper(self, mapping_matrix, mask):  # pragma: no cover
        return None

    with pytest.raises(TypeError, match="new_knob"):
        pci._assert_signature_covers(library, wrapper)


@pytest.mark.parametrize("name", list(pci.CANDIDATES))
def test__the_library_method_is_restored_after_the_context(name):
    from autoarray.operators.convolver import Convolver

    original = Convolver.convolved_mapping_matrix_from
    with pci.psf_convolution_injected(name):
        if name != "control":
            assert Convolver.convolved_mapping_matrix_from is not original
    assert Convolver.convolved_mapping_matrix_from is original

    with pytest.raises(RuntimeError, match="boom"):
        with pci.psf_convolution_injected(name):
            raise RuntimeError("boom")
    assert Convolver.convolved_mapping_matrix_from is original


def test__an_unknown_candidate_is_refused():
    with pytest.raises(ValueError, match="unknown PSF candidate"):
        with pci.psf_convolution_injected("fft_but_faster"):
            pass


def test__a_fresh_jit_inside_the_context_traces_the_candidate():
    """The patch acts at TRACE time: a jit traced inside the context carries it."""
    mask, kernel, mapping_matrix = _hst_like_fixture()
    convolver = aa.Convolver(kernel=kernel, use_fft=True)
    reference = _reference(kernel, mask, mapping_matrix)

    def fn(mm):
        return convolver.convolved_mapping_matrix_from(mapping_matrix=mm, mask=mask, xp=jnp)

    with pci.psf_convolution_injected("layout_src_first") as counts:
        got = np.asarray(jax.jit(fn)(jnp.asarray(mapping_matrix)))
    assert counts["jax"] == 1
    np.testing.assert_allclose(got, reference, rtol=0.0, atol=1e-12)


def test__provenance_records_the_shipped_and_used_frames():
    mask, kernel, _ = _hst_like_fixture()
    convolver = aa.Convolver(kernel=kernel, use_fft=True)
    shipped = pci.candidate_provenance("control", convolver, mask, n_src=50)
    big = pci.candidate_provenance("frame_pow2", convolver, mask, n_src=50)
    assert shipped["frame_used"] == shipped["fft_shape_shipped"]
    assert big["frame_used"] == list(pci.pow2_shape(big["fft_shape_shipped"]))
    assert all(s & (s - 1) == 0 for s in big["frame_used"])
    assert shipped["kernel_shape"] == [21, 21]
    assert shipped["cube_shape"] == shipped["frame_used"] + [50]
    first = pci.candidate_provenance("layout_src_first", convolver, mask, n_src=50)
    assert first["cube_shape"] == [50] + first["frame_used"]
    for name in pci.CANDIDATES:
        record = pci.candidate_provenance(name, convolver, mask, n_src=50)
        assert record["fp64_exact"] == pci.CANDIDATES[name].fp64_exact
        assert (record["patched"] is None) == (name == "control")


# ---------------------------------------------------------------------------
# The pre-registered phase-3 gate (psf_gate_rows): which rows carry gated=True
# ---------------------------------------------------------------------------

RTOL = 1.0e-9
LEVERS = [n for n, c in pci.CANDIDATES.items() if c.kind == "lever"]


def _fiducial(rel: float) -> list[dict]:
    return [{"pin": "route d == route b", "rel_diff": rel}]


def _draws(rel_b: float, rel_dc: float | None, n: int = 3) -> list[dict]:
    rows = []
    for k in range(n):
        row = {"draw": k, "draw_name": f"d{k}", "rel_diff": rel_b}
        if rel_dc is not None:
            row["rel_diff_vs_d_control"] = rel_dc
        rows.append(row)
    return rows


def test__gate__control_gates_the_fiducial_and_only_records_the_draws():
    fid, draws = _fiducial(1e-12), _draws(2.6e-9, None)
    gated, failed = pci.psf_gate_rows("control", fid, draws, RTOL)
    assert gated == fid and failed == []
    assert fid[0]["gated"] is True and fid[0]["gated_on"] == "rel_diff"
    assert fid[0]["status"] == "PASS"
    for row in draws:
        assert row["gated"] is False and row["gated_on"] is None
        assert row["status"] == "RECORDED"
        assert row["within_rtol_vs_b_library"] is False


def test__gate__control_fails_on_its_fiducial_pin():
    gated, failed = pci.psf_gate_rows("control", _fiducial(5e-9), _draws(0.0, None), RTOL)
    assert len(gated) == 1 and failed == ["route d == route b"]


@pytest.mark.parametrize("name", LEVERS)
def test__gate__a_lever_gates_the_draws_on_the_d_control_not_route_b(name):
    """The A100 solver residual (2.6e-9 vs route b) must not fail a lever whose
    convolution is exact against its d-control."""
    fid, draws = _fiducial(1e-12), _draws(2.6e-9, 1e-13)
    gated, failed = pci.psf_gate_rows(name, fid, draws, RTOL)
    assert failed == []
    assert len(gated) == 1 + len(draws)
    for row in draws:
        assert row["gated"] is True and row["gated_on"] == "rel_diff_vs_d_control"
        assert row["status"] == "PASS"
        assert row["within_rtol_vs_b_library"] is False


@pytest.mark.parametrize("name", LEVERS)
def test__gate__a_lever_fails_when_its_convolution_moves_the_answer(name):
    draws = _draws(1e-12, 3e-9)
    _, failed = pci.psf_gate_rows(name, _fiducial(1e-12), draws, RTOL)
    assert failed == [f"draw {k} (d{k}) vs d-control" for k in range(3)]
    assert all(r["status"] == "FAIL" for r in draws)


@pytest.mark.parametrize("name", LEVERS)
def test__gate__a_lever_still_gates_its_fiducial_against_route_b(name):
    _, failed = pci.psf_gate_rows(name, _fiducial(2e-9), _draws(0.0, 0.0), RTOL)
    assert failed == ["route d == route b"]


@pytest.mark.parametrize("name", LEVERS)
def test__gate__a_lever_draw_without_a_d_control_is_refused(name):
    with pytest.raises(AssertionError, match="rel_diff_vs_d_control"):
        pci.psf_gate_rows(name, _fiducial(0.0), _draws(0.0, None), RTOL)


@pytest.mark.parametrize("name", DIAGNOSTIC)
def test__gate__diagnostics_are_never_gated_and_never_pass(name):
    fid, draws = _fiducial(1e-5), _draws(1e-5, 1e-5)
    gated, failed = pci.psf_gate_rows(name, fid, draws, RTOL)
    assert gated == [] and failed == []
    for row in fid + draws:
        assert row["gated"] is False and row["status"] == "DIAGNOSTIC"


def test__gate__the_note_records_the_pre_registration():
    assert "Pre-registered before any A100 data" in pci.PSF_GATE_NOTE
    assert "344635" in pci.PSF_GATE_NOTE
    assert "1e-9" in pci.PSF_GATE_NOTE
