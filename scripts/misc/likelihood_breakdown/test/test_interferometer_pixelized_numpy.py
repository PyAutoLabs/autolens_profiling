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
    cells = {
        ("pixelization" if model == "rectangular" else model) + suffix
        for model, suffix in re.findall(
            r"scripts/interferometer/(\w+)/likelihood_breakdown(_numba)?\.py", text
        )
    }
    declared = set(re.findall(r"cell: interferometer/(\w+)/", text))
    assert cells and cells == declared, (
        f"every cell run needs its own WALL-BASIS row: {cells} vs {declared}"
    )


# --------------------------------------------------------------------------------------
# Phase 2 (autolens_profiling#332): the cached_property-aware counter, the sub-row pops,
# the adapt-image guard, the before/after record and the lever flags / submits.
# --------------------------------------------------------------------------------------


def _counted_class():
    from autonerves import cached_property

    class Inv:
        calls = {"cached": 0, "plain": 0}

        @cached_property
        def cached(self):
            Inv.calls["cached"] += 1
            return object()

        @property
        def plain(self):
            Inv.calls["plain"] += 1
            return 1.0

    return Inv


def test__counter_counts_computations_of_a_cached_property_and_restores_it():
    Inv = _counted_class()
    originals = dict(Inv.__dict__)

    def call():
        inv = Inv()
        first = inv.cached
        # A cache hit must not reach the counter, and must return the cached object.
        assert inv.cached is first
        # The autonerves CachedProperty caches under func.__name__: the wrapper must keep it.
        assert "cached" in inv.__dict__
        _ = (inv.plain, inv.plain)
        return 7

    out, counts = harness.count_evaluations([(Inv, "cached"), (Inv, "plain")], call)
    assert out == 7
    assert counts == {"cached": 1, "plain": 2}
    assert Inv.calls == {"cached": 1, "plain": 2}
    # Originals restored, same descriptor objects.
    assert Inv.__dict__["cached"] is originals["cached"]
    assert Inv.__dict__["plain"] is originals["plain"]


def test__counter_keeps_the_descriptor_type():
    Inv = _counted_class()
    counts = {"cached": 0, "plain": 0}
    assert type(harness.counting_descriptor(Inv.__dict__["cached"], counts, "cached")) is type(
        Inv.__dict__["cached"]
    )
    assert isinstance(harness.counting_descriptor(Inv.__dict__["plain"], counts, "plain"), property)


def test__counter_restores_the_originals_when_the_call_raises():
    Inv = _counted_class()
    original = Inv.__dict__["cached"]

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        harness.count_evaluations([(Inv, "cached")], boom)
    assert Inv.__dict__["cached"] is original


def test__library_sparse_classes_cache_f_and_d():
    """PyAutoArray >= e281abf3 (#582): the counter targets are cached descriptors."""
    from autoarray.inversion.inversion.interferometer.sparse import (
        InversionInterferometerSparse,
    )
    from autoarray.inversion.inversion.interferometer_numba.sparse import (
        InversionInterferometerSparseNumba,
    )

    for cls, name in (
        (InversionInterferometerSparse, "curvature_matrix_diag"),
        (InversionInterferometerSparseNumba, "curvature_matrix_diag"),
        (InversionInterferometerSparse, "data_vector"),
    ):
        descriptor = cls.__dict__[name]
        assert not isinstance(descriptor, property), f"{cls.__name__}.{name} is not cached"
        assert descriptor.func.__name__ == name


def test__recompute_drops_the_cached_value_every_time():
    Inv = _counted_class()
    inv = Inv()
    _ = inv.cached
    harness.recompute(inv, "cached")
    harness.recompute(inv, "cached")
    assert Inv.calls["cached"] == 3


def test__adapt_image_guard(tmp_path):
    # Missing at a non-preset radius: a hard error, never a silent regeneration.
    with pytest.raises(SystemExit, match="lensed_source.fits is missing"):
        harness.adapt_image_guard(tmp_path, 5.0, 3.5)
    # Missing at the preset radius: the documented first-run regeneration, recorded.
    rec = harness.adapt_image_guard(tmp_path, 3.5, 3.5)
    assert rec["cache_existed"] is False and rec["md5_before"] is None
    (tmp_path / "lensed_source.fits").write_bytes(b"abc")
    rec = harness.adapt_image_guard(tmp_path, 5.0, 3.5)
    assert rec["cache_existed"] is True
    assert rec["md5_before"] == "900150983cd24fb0d6963f7d28e17f72"


def test__previous_row_carries_the_replaced_rows_numbers(tmp_path):
    import json

    assert harness.previous_row(tmp_path / "missing.json") is None
    path = tmp_path / "row.json"
    path.write_text(
        json.dumps(
            {
                "arms": {
                    "numba": {"full_call": {"mean_s": 2.0}},
                    "numpy_fft": {"full_call": {"mean_s": 4.0}},
                    "jax_cpu_fft": {"path": "x"},
                },
                "evaluations_per_figure_of_merit": {"curvature_matrix_diag": 2, "data_vector": 2},
                "figure_of_merit_reference": -1.0,
                "configuration": {"nnz_per_source_column": 29.1},
                "source_revisions": {"PyAutoArray": "abc"},
                "agreement": {"numba_over_numpy_fft_full_call": 0.5},
            }
        )
    )
    prev = harness.previous_row(path)
    assert prev["full_call_mean_s"] == {"numba": 2.0, "numpy_fft": 4.0}
    assert prev["evaluations_per_figure_of_merit"]["curvature_matrix_diag"] == 2
    assert prev["nnz_per_source_column"] == 29.1
    assert prev["numba_over_numpy_fft_full_call"] == 0.5


def test__lever_flags_parse():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    harness.add_cell_args(parser)
    args = parser.parse_args([])
    assert args.levers == "" and args.lever_threads == "1,2,4"
    args = parser.parse_args(["--levers", "threads,memo", "--lever-threads", "1,2"])
    assert (args.levers, args.lever_threads) == ("threads,memo", "1,2")
    assert harness.LEVERS == ("threads", "memo", "logdet", "marshal")


CROSSOVER_SUBMITS = sorted(
    BATCH_CPU.glob("submit_breakdown_interferometer_*_numba_ral_crossover_*")
)
LEVER_SUBMITS = sorted(BATCH_CPU.glob("submit_breakdown_interferometer_numba_levers_ral_*"))


def test__phase_2_submits_exist():
    names = {p.name for p in CROSSOVER_SUBMITS}
    for cell in ("delaunay", "pixelization"):
        assert f"submit_breakdown_interferometer_{cell}_numba_ral_crossover_fp64" in names
    assert LEVER_SUBMITS, "the lever submit is missing"


@pytest.mark.parametrize("path", CROSSOVER_SUBMITS, ids=lambda p: p.name)
def test__crossover_submit_sweeps_radii_at_the_default_names(path):
    text = path.read_text()
    # r3.5 re-runs keep the phase-1 names (the before/after record); other radii are suffixed.
    radii = re.search(r"^RADII=\((.*?)\)", text, re.M).group(1).split()
    instruments = re.search(r"^INSTRUMENTS=\((.*?)\)", text, re.M).group(1).split()
    assert len(radii) == len(instruments)
    array = re.search(r"^#SBATCH --array=0-(\d+)", text, re.M)
    assert array and int(array.group(1)) == len(radii) - 1
    pairs = set(zip(instruments, radii))
    for pair in (("sma", "3.5"), ("alma", "2.0"), ("alma", "3.5"), ("alma", "5.0")):
        assert pair in pairs
    assert ("alma_high", "3.5") in pairs
    assert "--levers" not in text


@pytest.mark.parametrize("path", LEVER_SUBMITS, ids=lambda p: p.name)
def test__lever_submit_is_a_cpu_job_whose_only_extra_thread_is_the_numba_pool(path):
    text = path.read_text()
    assert re.search(r"^#SBATCH --partition=gpu$", text, re.M)
    assert not re.search(r"^#SBATCH --gres", text, re.M)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"export {var}=1" in text
    pool = int(re.search(r"export AUTOLENS_PROFILING_LEVER_NUMBA_THREADS=(\d+)", text).group(1))
    cpus = int(re.search(r"^#SBATCH --cpus-per-task=(\d+)", text, re.M).group(1))
    threads = [int(t) for t in re.search(r"--lever-threads (\S+)", text).group(1).split(",")]
    assert max(threads) <= pool <= cpus, "the pool must fit the allocation"
    configs = re.findall(r"--config-name (\S+)", text)
    assert configs and all(c.startswith("hpc_ral_cpu_fp64_levers") for c in configs)
    cells = {
        ("pixelization" if model == "rectangular" else model) + suffix
        for model, suffix in re.findall(
            r"scripts/interferometer/(\w+)/likelihood_breakdown(_numba)?\.py", text
        )
    }
    declared = set(re.findall(r"cell: interferometer/(\w+)/", text))
    assert cells and cells == declared
