"""The datacube and multi-dataset standard-size drivers (likelihood-lifecycle 2f, #475).

Decision D2 fixes the cube (channel 0 is 2b's cell, a Gaussian emission line scales the rest),
D3 the cube reference (derived from one channel reference's terms), D4 the breakdown projection,
D6 the multi-dataset scales and D8 its tolerance. Fixtures only: no dataset is loaded and no cell
script is run (the scripts are read as text).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
import parity_witness  # noqa: E402
from likelihood_breakdown import datacube_standard as dc  # noqa: E402
from likelihood_breakdown import multi_dataset_standard as md  # noqa: E402

#: The step names of 2b's JAX W~ harness (a 2026-10-28 sma run) and its dense arm.
JAX_STEPS = {
    "Ray-trace grids (data + mesh vertices)": 0.005,
    "Border relocation + mesh interpolation (mapper)": 0.29,
    "Mapping matrix L": 0.12,
    "Regularization matrix H": 0.25,
    "Sparse triplets (extent grid)": 0.001,
    "Data vector D = Lᵀ d~": 0.13,
    "Curvature matrix F = Aᵀ W~ A (10 blocks of 128)": 1.55,
    "Reconstruction (pdip)": 1.78,
    "Log-det terms (2 Choleskys)": 0.22,
    "Fast chi-squared + figure of merit": 0.005,
}
DENSE_ONLY = {
    "Transformed mapping matrix T (NUFFT, 50-column chunks)": 9.0,
    "Data vector D (dense, Re + Im)": 0.003,
    "Curvature matrix F (dense Tᵀ N⁻¹ T, Re + Im)": 0.06,
}
#: The step names of 2b's numba harness (a 2026-10-28 sma run).
NUMBA_STEPS = {
    "Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.06,
    "Mapping matrix L": 0.005,
    "Curvature F + H: F by the packaged route (cost model: numba or NumPy FFT)": 0.2,
    "F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.01,
    "Regularization matrix H": 0.05,
    "Reconstruction: D = Lᵀ d~ + fnnls": 0.5,
    "Log-det (F + H) (fnnls Cholesky reuse)": 0.1,
    "Log-det H": 0.1,
    "Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)": 0.001,
    "Regularization term sᵀHs": 0.001,
    "Noise normalisation + evidence assembly": 0.001,
}
INVARIANT = {
    "Inversion build: ray-trace + mesh + mapper (+ routing gate)",
    "Curvature F + H: F by the packaged route (cost model: numba or NumPy FFT)",
    "F marshalling: kernel_index_arrays (CSR / CSC / extent)",
    "Ray-trace grids (data + mesh vertices)",
    "Border relocation + mesh interpolation (mapper)",
    "Mapping matrix L",
    "Sparse triplets (extent grid)",
    "Curvature matrix F = Aᵀ W~ A (10 blocks of 128)",
    "Curvature matrix F (dense Tᵀ N⁻¹ T, Re + Im)",
}


# ------------------------------------------------------------------------------ D2 cube


def test_channel_scales_are_an_emission_line_peaking_at_the_2b_cell():
    scales = dc.channel_scales(dc.N_CHANNELS)
    assert dc.N_CHANNELS == 34 and len(scales) == 34
    assert scales[0] == 1.0  # channel 0 IS the 2b interferometer cell
    assert all(a > b > 0 for a, b in zip(scales, scales[1:]))
    assert scales[-1] == pytest.approx(math.exp(-0.5 * (33 / 17) ** 2))
    assert dc.channel_scales(1) == [1.0]


# ------------------------------------------------------------------------------ D3 reference


def test_cube_reference_scales_chi_squared_and_regularization_only():
    ref = {
        "log_evidence": -0.5 * (100.0 + 20.0 + 7.0 - 3.0 + 50.0),
        "chi_squared": 100.0,
        "regularization_term": 20.0,
        "log_det_curvature_reg": 7.0,
        "log_det_regularization": 3.0,
        "noise_normalization": 50.0,
    }
    cube = dc.cube_reference_from(ref, [1.0, 0.5])
    assert cube["log_evidence_per_channel"][0] == pytest.approx(ref["log_evidence"])
    assert cube["log_evidence_per_channel"][1] == pytest.approx(-0.5 * (0.25 * 120.0 + 54.0))
    assert cube["log_evidence"] == pytest.approx(sum(cube["log_evidence_per_channel"]))
    assert cube["n_channels"] == 2 and cube["channel_scales"] == [1.0, 0.5]


def _row(tmp_path, revisions, unit_vector, scales):
    cell = tmp_path / "results/datacube/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100"
    cell.mkdir(parents=True)
    candidate = {
        "log_likelihood": -10.0,
        "quantity": dc.PARITY_QUANTITY,
        "instance_unit_vector": unit_vector,
        "source_pixels": 1280,
        "source_pixels_requested": 1250,
        "use_mixed_precision": False,
        "channel_scales": scales,
        "n_channels": len(scales),
    }
    row = {
        "autolens_version": "2026.8.17.1",
        "device": {"provenance": {"library_revisions": revisions}},
        "parity_candidate": candidate,
    }
    (cell / "runtime.json").write_text(json.dumps(row))
    return cell


def test_finalise_parity_rewrites_only_rows_of_the_reference_instance(tmp_path):
    revisions = {"PyAutoArray": "abc"}
    reference = {
        "log_evidence": -4.0,
        "chi_squared": 6.0,
        "regularization_term": 2.0,
        "log_det_curvature_reg": 1.0,
        "log_det_regularization": 1.0,
        "noise_normalization": 0.0,
        "instance_unit_vector": [0.5, 0.5],
        "library_revisions": revisions,
        "method_short": "chunked",
    }
    # -0.5 * (8 + 0) + -0.5 * (0.25 * 8 + 0) = -4 - 1 = -5, so a candidate of -5 matches.
    cell = _row(tmp_path, revisions, [0.5, 0.5], [1.0, 0.5])
    row = json.loads((cell / "runtime.json").read_text())
    row["parity_candidate"]["log_likelihood"] = -5.0
    (cell / "runtime.json").write_text(json.dumps(row))
    out = dc.finalise_parity("alma", results_root=tmp_path, reference=reference)
    assert [Path(r["row"]).name for r in out["finalised"]] == ["runtime.json"]
    witness = json.loads((cell / "parity.json").read_text())
    assert witness["status"] == "match" and parity_witness.validate(witness) == []
    assert witness["reference_log_likelihood"] == pytest.approx(-5.0)
    assert witness["tolerance"] == pytest.approx(1e-8 * 5.0)  # 2b's R7 (D3)

    other = dc.finalise_parity(
        "alma", results_root=tmp_path, reference={**reference, "instance_unit_vector": [0.1]}
    )
    assert other["failed"] and not other["finalised"]
    assert dc.finalise_parity("alma", results_root=tmp_path, reference=None)["failed"]


# ------------------------------------------------------------------------------ D4 projection


def test_step_classification_matches_the_shared_preloads_path():
    for name in {**JAX_STEPS, **DENSE_ONLY, **NUMBA_STEPS}:
        assert dc.step_is_invariant(name) == (name in INVARIANT), name


def test_cube_projection_counts_invariant_steps_once():
    row = {"steps": JAX_STEPS}
    proj = dc.cube_projection(row, 34)
    inv = sum(v for k, v in JAX_STEPS.items() if k in INVARIANT)
    var = sum(v for k, v in JAX_STEPS.items() if k not in INVARIANT)
    assert proj["channel_invariant_s"] == pytest.approx(inv)
    assert proj["total_step_by_step_cube"] == pytest.approx(inv + 34 * var)
    assert set(proj["channel_invariant_steps"]) | set(proj["channel_variant_steps"]) == set(
        JAX_STEPS
    )
    # The dense route's T is per channel (D = Tᵀ N⁻¹ d needs it in every channel).
    dense = dc.cube_projection({"steps": {**DENSE_ONLY}}, 2)
    assert (
        "Transformed mapping matrix T (NUFFT, 50-column chunks)" in dense["channel_variant_steps"]
    )


# ------------------------------------------------------------------------------ D6 / D8


def test_multi_dataset_scales_and_cell():
    assert md.N_DATASETS == 2 and md.dataset_scales(2) == [1.0, 0.5]
    assert md.dataset_scales(1) == [1.0]
    assert md.dataset_scales(3)[2] == 0.25
    assert md.INSTRUMENT == "hst" and md.VARIANT == "delaunay"


# ------------------------------------------------------------------------------ scripts


@pytest.mark.parametrize(
    "script, needle",
    [
        ("scripts/datacube/delaunay/likelihood_runtime.py", "datacube_standard import run_jax"),
        ("scripts/datacube/delaunay/likelihood_runtime_numba.py", "import run_numba"),
        ("scripts/datacube/delaunay/likelihood_breakdown.py", "datacube_standard.run_breakdown"),
        (
            "scripts/datacube/delaunay/likelihood_breakdown_numba.py",
            "datacube_standard.run_breakdown",
        ),
        ("scripts/datacube/delaunay/parity_reference.py", "finalise_parity"),
        ("scripts/multi_dataset/delaunay/likelihood_runtime.py", "import run_jax"),
        ("scripts/multi_dataset/delaunay/likelihood_runtime_numba.py", "import run_numba"),
        ("scripts/multi_dataset/delaunay/likelihood_breakdown.py", "import run_breakdown"),
        ("scripts/multi_dataset/delaunay/likelihood_breakdown_numba.py", "import run_breakdown"),
    ],
)
def test_cell_scripts_hand_off_to_the_drivers_and_keep_the_smoke_exit(script, needle):
    text = (ROOT / script).read_text()
    assert needle in text
    assert 'AUTOLENS_PROFILING_SMOKE") == "1"' in text


def test_numba_scripts_pin_threads_before_numpy():
    for script in (
        "scripts/datacube/delaunay/likelihood_runtime_numba.py",
        "scripts/datacube/delaunay/likelihood_breakdown_numba.py",
        "scripts/multi_dataset/delaunay/likelihood_runtime_numba.py",
        "scripts/multi_dataset/delaunay/likelihood_breakdown_numba.py",
    ):
        text = (ROOT / script).read_text()
        assert "_pin_thread_env(1)" in text
        assert text.index("_pin_thread_env(1)") < text.index("from likelihood_breakdown"), script
