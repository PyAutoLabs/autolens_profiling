"""The interferometer standard-size cell and its parity witness (likelihood-lifecycle 2b, #469).

Ruling R3 fixes the cell (the hst ``source_pix[2]`` mesh and regularization at the standard
size, no lens light, the truth lensed-source adapt image uncapped), R7 the witness tolerance
(1e-8 x |reference| in fp64, 1e-3 mixed precision) and R4 the numba headline (the packaged
cost-model route, forced routes as sub-rows). Fixtures only: no dataset is loaded and no cell
script is run (the scripts are read as text / AST).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
import parity_witness  # noqa: E402

import _production_config as prod  # noqa: E402
import _profile_cli as pc  # noqa: E402

VARIANTS = ("rectangular", "delaunay")


# ------------------------------------------------------------------------------ R3 preset


@pytest.mark.parametrize("variant", VARIANTS)
def test_interferometer_preset_is_r3(variant):
    preset = prod.interferometer_standard_preset(variant)
    base = prod.HST_SOURCE_PIX_2 if variant == "delaunay" else prod.HST_RECT_ADAPT
    assert preset.dataset_type == "interferometer"
    assert preset.adapt_snr_cap is None  # the truth lensed source, no S/N cap
    assert preset.mge_total_gaussians == 0 and preset.positions_factor is None
    assert preset.regularization == base.regularization  # AdaptSplit / Adapt, free
    assert preset.size_override  # no imaging production witness or pin applies
    record = preset.as_json()
    assert record["lens_light"] is None and record["positions_penalty"] is None
    assert record["over_sample_size_lp_rule"] is None
    assert "no S/N cap" in record["adapt_image"]
    if variant == "delaunay":
        assert (preset.hilbert_pixels, preset.edge_pixels) == (1250, 30)
        assert (preset.hilbert_weight_power, preset.hilbert_weight_floor) == (3.5, 0.01)
        assert record["mesh"]["vertices"] == 1280 and record["regularization"]["free"]
        assert preset.regularization == "adapt_split"
    else:
        assert preset.rect_pixels_yx == 35 and record["mesh"]["source_pixels"] == 1225
        assert record["mesh"]["class"] == "RectangularBilinearAdaptImage"
        assert record["mesh"]["free_weights"] == prod.PRODUCTION_RECT_WEIGHT_PRIORS
        assert preset.regularization == "adapt"


def test_interferometer_preset_refuses_other_variants_and_leaves_imaging_alone():
    with pytest.raises(KeyError):
        prod.interferometer_standard_preset("mge")
    # The imaging presets keep their imaging record (no interferometer fields leak in).
    assert prod.HST_SOURCE_PIX_2.dataset_type == "imaging"
    assert "dataset_type" not in prod.HST_SOURCE_PIX_2.as_json()
    assert prod.HST_SOURCE_PIX_2.as_json()["lens_light"].startswith("mge_")


def test_iid_unit_vectors_are_the_iid_instances_draws():
    preset = prod.interferometer_standard_preset("rectangular")
    a = prod.iid_unit_vectors(3, 7, preset)
    b = prod.iid_unit_vectors(3, 7, preset)
    assert a == b and len(a) == 3 and all(len(u) == 7 for u in a)
    assert all(preset.iid_unit_low <= x <= preset.iid_unit_high for u in a for x in u)


# ------------------------------------------------------------------------- R7 tolerance


def test_parity_tolerance_is_1e8_for_the_interferometer_and_unchanged_for_imaging():
    assert pc.parity_rtol("interferometer", "fp64") == 1e-8
    assert pc.parity_rtol("interferometer", "mixed") == 1e-3
    assert pc.parity_rtol("imaging", "fp64") == pc.PARITY_RTOL["fp64"] == 1e-4
    assert pc.parity_rtol(None, "fp64") == 1e-4
    ref = -3.0e8  # about jvla's |log L|
    common = dict(
        reference_log_likelihood=ref,
        use_mixed_precision=False,
        reference_source="test",
        autolens_version="2026.10.25.1",
    )
    inside = pc.parity_witness_doc(
        log_likelihood=ref + 2.9, dataset_type="interferometer", **common
    )
    outside = pc.parity_witness_doc(
        log_likelihood=ref + 3.1, dataset_type="interferometer", **common
    )
    assert inside["status"] == "match" and outside["status"] == "diverged"
    assert inside["tolerance"] == pytest.approx(3.0)
    assert "1e-08" in inside["tolerance_rule"]
    # The imaging rule is untouched: the same gap is a match at 1e-4.
    assert pc.parity_witness_doc(log_likelihood=ref + 3.1, **common)["status"] == "match"
    for doc in (inside, outside):
        assert parity_witness.validate(doc) == []


def test_a_pending_reference_is_a_valid_not_assessed_witness_with_its_reason():
    doc = pc.parity_witness_doc(
        log_likelihood=-1.0,
        reference_log_likelihood=float("nan"),
        use_mixed_precision=False,
        reference_source="visibility-chunked eager NumPy dense reference (reference task pending)",
        autolens_version="2026.10.25.1",
        dataset_type="interferometer",
        pending_reason="reference pending: no cached reference",
    )
    assert doc["status"] == "not_assessed" and doc["reason"].startswith("reference pending")
    assert parity_witness.validate(doc) == []


# ------------------------------------------------------------------ the reference cache key


def test_reference_cache_is_keyed_on_dataset_revisions_and_instance(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("data.fits", "noise_map.fits", "uv_wavelengths.fits", "lensed_source.fits"):
        (dataset / name).write_bytes(name.encode())
    fingerprint = prod.dataset_fingerprint(dataset)
    assert all(fingerprint.values())
    doc = {
        "schema": "p2b-interferometer-reference",
        "version": 1,
        "log_evidence": -12.5,
        "dataset_fingerprint": fingerprint,
        "library_revisions": {"PyAutoArray": "abc"},
        "instance_unit_vector": [0.5, 0.45],
    }
    path = prod.interferometer_reference_path(tmp_path, "delaunay", "jvla")
    assert path == tmp_path / "output/p2b_references/interferometer_delaunay_jvla.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc))
    kw = dict(dataset_path=dataset, library_revisions={"PyAutoArray": "abc"})
    got, why = prod.load_interferometer_reference(
        tmp_path, "delaunay", "jvla", unit_vector=[0.5, 0.45], **kw
    )
    assert got["log_evidence"] == -12.5 and why == "ok"
    assert (
        prod.load_interferometer_reference(
            tmp_path, "delaunay", "jvla", unit_vector=[0.5, 0.46], **kw
        )[0]
        is None
    )
    assert (
        prod.load_interferometer_reference(
            tmp_path,
            "delaunay",
            "jvla",
            unit_vector=[0.5, 0.45],
            dataset_path=dataset,
            library_revisions={"PyAutoArray": "def"},
        )[0]
        is None
    )
    (dataset / "data.fits").write_bytes(b"changed")
    assert (
        prod.load_interferometer_reference(
            tmp_path, "delaunay", "jvla", unit_vector=[0.5, 0.45], **kw
        )[0]
        is None
    )


# ---------------------------------------------------------------- the scripts (text / AST)


def _source(rel):
    return (ROOT / rel).read_text()


@pytest.mark.parametrize("variant", VARIANTS)
def test_jax_runtime_scripts_hand_standard_size_to_the_shared_driver(variant):
    text = _source(f"scripts/interferometer/{variant}/likelihood_runtime.py")
    assert "if _cli.standard_size:" in text
    assert f'run_jax("{variant}", _cli' in text
    # The driver runs before the historic fiducial cell builds anything.
    assert text.index("run_jax(") < text.index("al.Interferometer.from_fits(")


@pytest.mark.parametrize("variant", VARIANTS)
def test_numba_runtime_scripts_pin_threads_before_numpy_and_run_the_driver(variant):
    text = _source(f"scripts/interferometer/{variant}/likelihood_runtime_numba.py")
    tree = ast.parse(text)
    imports = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]  # module-level imports, in order
    names = [getattr(n, "module", None) or n.names[0].name for n in imports]
    assert "numpy" not in names and "autolens" not in names  # the driver imports them
    assert text.index("_pin_thread_env(1)") < text.index("run_numba(")
    assert f'run_numba("{variant}", _cli, _thread_env' in text


def test_numba_driver_headline_is_the_packaged_route_with_forced_sub_rows():
    from likelihood_breakdown import interferometer_standard as std

    assert std.NUMBA_ARMS["default"] == {}
    assert std.NUMBA_ARMS["numba"] == {"interferometer_route": "numba"}
    assert std.NUMBA_ARMS["numpy_fft"] == {"interferometer_route": "fft"}
    assert std._numba_parser().parse_known_args([])[0].arms == "default,numba,numpy_fft"
    numpy_harness = _source("scripts/misc/likelihood_breakdown/interferometer_pixelized_numpy.py")
    assert 'STANDARD_ARMS = "default,numba,numpy_fft"' in numpy_harness
    assert '"default": None' in numpy_harness


def test_breakdown_harness_has_the_dense_primary_mode_and_the_standard_cell():
    text = _source("scripts/misc/likelihood_breakdown/interferometer_pixelized.py")
    assert '"--dense"' in text and "dense_primary" in text
    assert "pc.standard_interferometer_cell(" in text
    assert '"resources": resources_dict("jax")' in text
    numpy_harness = _source("scripts/misc/likelihood_breakdown/interferometer_pixelized_numpy.py")
    assert "pc.standard_interferometer_cell(" in numpy_harness
    assert '"resources": resources_dict("numba")' in numpy_harness


def test_dense_flag_parses_on_the_breakdown_cell_parser():
    import argparse

    from likelihood_breakdown import interferometer_pixelized as harness

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    harness.add_cell_args(parser)
    assert parser.parse_known_args(["--dense"])[0].dense
    assert not parser.parse_known_args([])[0].dense
