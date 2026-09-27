"""Static checks on the interferometer library-dispatch numba cells (autolens_profiling#326).

Covers what the harness decides without running a likelihood:

- the ``--mask-radius`` naming rule: the preset radius (3.5") keeps every existing output
  name, any other radius appends ``_r<radius>``, for the JAX cells and the numba cells
  alike, so a phase-2 / phase-3 mask sweep can never overwrite an r3.5 row;
- the arm -> inversion-class table and the gate values the arms are driven by (the numba
  arm forced open, the FFT arm at the kill switch);
- the cell-local flags parse through the shared staged CLI;
- the RAL CPU submits: ``gpu`` partition with no ``--gres``, one thread pinned, every leg
  config-tagged under ``hpc_ral_cpu_fp64``, and a WALL-BASIS row for every cell they run.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_interferometer_pixelized_numpy.py
"""

from __future__ import annotations

import argparse
import re
import sys as _sys
from pathlib import Path as _Path
from types import SimpleNamespace

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

from likelihood_breakdown import interferometer_pixelized as shared  # noqa: E402
from likelihood_breakdown import interferometer_pixelized_numpy as harness  # noqa: E402

BATCH_CPU = ROOT / "hpc" / "batch_cpu"
SUBMITS = sorted(BATCH_CPU.glob("submit_breakdown_interferometer_*_numba_ral_*"))


def _cli(tmp_path, config_name=None, source_pixels=None, regularization=None):
    return SimpleNamespace(
        config_name=config_name,
        output_dir=None,
        source_pixels=source_pixels,
        use_sparse_operator=False,
        rect_mesh="bilinear",
        regularization=regularization,
    )


def test__mask_radius_suffix_is_empty_only_at_the_preset_radius():
    assert shared.mask_radius_suffix(3.5, 3.5) == ""
    assert shared.mask_radius_suffix(2.0, 3.5) == "_r2.0"
    assert shared.mask_radius_suffix(5, 3.5) == "_r5.0"


@pytest.mark.parametrize(
    "instrument, config_name, suffix, expected",
    [
        (
            "sma",
            None,
            "",
            "results/breakdown/interferometer/delaunay_numba_breakdown_sma_v1.2.json",
        ),
        (
            "sma",
            None,
            "_r2.0",
            "results/breakdown/interferometer/delaunay_numba_breakdown_sma_v1.2_r2.0.json",
        ),
        (
            "sma",
            "hpc_ral_cpu_fp64",
            "",
            "results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64.json",
        ),
        (
            "alma",
            "hpc_ral_cpu_fp64",
            "",
            "results/breakdown/interferometer/delaunay_numba_hpc_ral_cpu_fp64.json",
        ),
        (
            "alma_high",
            "hpc_ral_cpu_fp64",
            "_r5.0",
            "results/breakdown/interferometer/alma_high/delaunay_numba_hpc_ral_cpu_fp64_r5.0.json",
        ),
    ],
)
def test__result_paths_numba_cell(tmp_path, instrument, config_name, suffix, expected):
    json_path, png_path = shared.result_paths(
        _cli(tmp_path, config_name=config_name),
        tmp_path,
        instrument,
        "delaunay_numba",
        "1.2",
        suffix,
    )
    assert json_path == tmp_path / expected
    assert png_path == json_path.with_suffix(".png")


def test__result_paths_jax_cell_keeps_its_pre_326_names(tmp_path):
    json_path, _ = shared.result_paths(
        _cli(tmp_path, config_name="hpc_a100_fp64"), tmp_path, "sma", "delaunay", "1.2"
    )
    assert (
        json_path == tmp_path / "results/breakdown/interferometer/sma/delaunay_hpc_a100_fp64.json"
    )

    json_path, _ = shared.result_paths(
        _cli(tmp_path, regularization="constant_split"), tmp_path, "alma", "delaunay", "1.2"
    )
    assert json_path.name == "delaunay_breakdown_alma_v1.2_constant_split.json"


def test__arms_map_to_the_library_inversion_classes():
    assert harness.ARM_CLASSES == {
        "numba": "InversionInterferometerSparseNumba",
        "numpy_fft": "InversionInterferometerSparse",
    }
    # The numba arm must be admitted whatever the geometry (alma_high Delaunay is ~123
    # non-zeros per column, twice the packaged gate).
    assert harness.NUMBA_GATE_FORCED > 1e6
    assert harness.STEP_SUM_BAND == (0.9, 1.1)
    assert harness.AGREEMENT_BAR_NATS == 0.5


def test__cell_flags_parse():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    harness.add_cell_args(parser)
    args = parser.parse_args([])
    assert args.arms == "numba,numpy_fft"
    assert args.numba_gate == harness.NUMBA_GATE_FORCED
    assert args.mask_radius is None
    assert args.preload_cache == "on"

    args = parser.parse_args(["--mask-radius", "2.0", "--arms", "numba", "--numba-gate", "60"])
    assert (args.mask_radius, args.arms, args.numba_gate) == (2.0, "numba", 60.0)

    jax_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    shared.add_cell_args(jax_parser)
    assert jax_parser.parse_args(["--mask-radius", "5.0"]).mask_radius == 5.0


def test__the_submits_exist():
    names = {p.name for p in SUBMITS}
    for cell in ("delaunay", "pixelization"):
        for instrument in ("sma", "alma", "alma_high"):
            assert f"submit_breakdown_interferometer_{cell}_numba_ral_{instrument}_fp64" in names
        assert f"submit_breakdown_interferometer_{cell}_numba_ral_alma_fp64_n_sweep" in names


@pytest.mark.parametrize("path", SUBMITS, ids=lambda p: p.name)
def test__submit_is_a_quiet_single_thread_cpu_job(path):
    text = path.read_text()
    assert re.search(r"^#SBATCH --partition=gpu$", text, re.M)
    assert not re.search(r"^#SBATCH --gres", text, re.M), "a CPU timing leg must not hold a GPU"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        assert f"export {var}=1" in text
    assert "export JAX_PLATFORMS=cpu" in text
    assert "intra_op_parallelism_threads=1" in text
    configs = re.findall(r"--config-name (\S+)", text)
    assert configs and all(c.startswith("hpc_ral_cpu_fp64") for c in configs)
    cells = set(re.findall(r"scripts/interferometer/likelihood_breakdown/(\w+)\.py", text))
    declared = set(re.findall(r"cell: interferometer/(\w+)/", text))
    assert cells and cells == declared, (
        f"every cell run needs its own WALL-BASIS row: {cells} vs {declared}"
    )
