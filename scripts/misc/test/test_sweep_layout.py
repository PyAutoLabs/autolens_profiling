"""``likelihood_runtime/sweep_layout.py``: sweep rows and comparisons, one rule for writer and reader.

The sweep writes each row to ``placed_row(...)`` (passed as ``--output-path``) and
``aggregate.py`` reads the same path, so no reader rebuilds a location from partial hints and
no two cells of the matrix share a file (autolens_profiling#442 PR 3b review finding 1).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MISC = Path(__file__).resolve().parents[1]
ROOT = _MISC.parents[1]
for _p in (ROOT, _MISC, _MISC / "likelihood_runtime", _MISC / "tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


layout = _load("sweep_layout", _MISC / "likelihood_runtime" / "sweep_layout.py")
aggregate = _load("aggregate_layout", _MISC / "likelihood_runtime" / "aggregate.py")
sweep = _load("sweep_layout_driver", _MISC / "likelihood_runtime" / "sweep.py")
import catalogue_lifecycle as lc  # noqa: E402

MATRIX = [
    (cls, model, inst, cfg, sparse)
    for cls, model, insts in layout.CELLS
    for inst in insts
    for cfg in layout.CONFIG_NAMES
    for sparse in (False, True)
]
#: The committed interferometer/datacube cells the review found did not round-trip.
INTERFEROMETRIC_CELLS = [
    ("interferometer", "delaunay", "sma"),
    ("interferometer", "pixelization", "sma"),
    ("datacube", "delaunay", "sma"),
]


def test_every_matrix_row_has_its_own_mirrored_path_under_its_instrument():
    paths = [layout.placed_row(*row) for row in MATRIX]
    assert len(set(paths)) == len(paths), "two sweep rows share a file"
    for (cls, _model, inst, _cfg, sparse), path in zip(MATRIX, paths):
        facts = lc.mirrored(path.relative_to(ROOT).as_posix())
        assert facts is not None, f"{path} is not a mirrored cell"
        assert facts["dataset_type"] == cls and facts["instrument"] == inst
        assert (facts["setup"]["operator"] == "sparse") == sparse
        if cls in layout.INTERFEROMETRIC and not sparse:
            assert facts["setup"]["transformer"] == "nufft"


def test_aggregate_reads_exactly_the_path_the_sweep_writes(monkeypatch):
    written = {layout.placed_row(*row): row for row in MATRIX}
    monkeypatch.setattr(aggregate, "is_fresh", lambda p: Path(p) in written)
    for cls, model, insts in layout.CELLS:
        for inst in insts:
            files = aggregate._cell_files(layout.LEGACY_RUNTIME_ROOT / cls / model / inst)
            for cfg in layout.CONFIG_NAMES:
                for sparse in (False, True):
                    name = layout.row_name(model, cfg, sparse)
                    assert files[name] == layout.placed_row(cls, model, inst, cfg, sparse)


def test_the_sweep_passes_the_placed_path_as_output_path(capsys):
    config = sweep.CONFIGS[0]
    for cls, model, inst in INTERFEROMETRIC_CELLS + [("imaging", "delaunay_nn", "hst")]:
        script = sweep.runtime_path(cls, model, ROOT)
        out_dir = layout.LEGACY_RUNTIME_ROOT / cls / model / inst
        sweep._run_one("python", script, config, out_dir, dry_run=True, instrument=inst)
        cmd = capsys.readouterr().out
        expected = layout.placed_row(cls, model, inst, config.name, False)
        assert f"--output-path {expected}" in cmd and "--output-dir" not in cmd


def test_a_verbatim_output_root_keeps_legacy_names(tmp_path):
    row = layout.placed_row("interferometer", "delaunay", "sma", "local_cpu_fp64", False, tmp_path)
    assert row == tmp_path / "interferometer/delaunay/sma/delaunay_local_cpu_fp64.json"


@pytest.mark.parametrize("cell", INTERFEROMETRIC_CELLS)
def test_committed_interferometric_cells_round_trip(cell):
    cell_dir = layout.LEGACY_RUNTIME_ROOT.joinpath(*cell)
    comparison = aggregate._aggregate_cell(cell_dir)
    dest = layout.placed_comparison(cell_dir, aggregate._rows_instrument(comparison))
    committed = json.loads(dest.read_text())["configs"]
    rebuilt = json.loads(json.dumps(comparison["configs"], default=str))
    assert rebuilt == committed


def test_a_two_level_cell_comparison_lands_under_its_rows_instrument():
    cell_dir = layout.LEGACY_RUNTIME_ROOT / "imaging" / "delaunay_nn"
    comparison = aggregate._aggregate_cell(cell_dir)
    assert aggregate._rows_instrument(comparison) == "hst"
    dest = layout.placed_comparison(cell_dir, "hst")
    assert dest == ROOT / "results/imaging/delaunay_nn/hst/comparison.json"


def test_resolve_output_paths_honours_output_path_and_names_the_instrument(tmp_path):
    import _profile_cli as pc

    exact = tmp_path / "x" / "runtime.json"
    cli = pc.parse_profile_cli(
        argv=["--config-name", "local_cpu_fp64", "--output-path", str(exact)]
    )
    json_path, png_path = pc.resolve_output_paths(cli, ROOT / "results/runtime/a/b", "b_x")
    assert (json_path, png_path) == (exact, exact.with_suffix(".png"))
    legacy = ROOT / "results" / "runtime" / "interferometer" / "delaunay"
    assert pc.runtime_instrument_dir(legacy, "sma") == legacy / "sma"
    assert pc.runtime_instrument_dir(legacy / "sma", "sma") == legacy / "sma"
    assert pc.runtime_instrument_dir(tmp_path, "sma") == tmp_path
