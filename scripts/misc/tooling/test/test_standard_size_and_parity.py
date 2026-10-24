"""--standard-size and the parity.json writer (likelihood-lifecycle 2a, autolens_profiling#456).

``--standard-size`` gives each cell its variant's standard source-pixel count (Delaunay 1250,
rectangular 35x35) for every instrument, numba presets included, without the ``--source-pixels``
scan suffix; rows record ``source_pixels_requested`` and the catalogue's size rule matches on it.
``write_parity_json`` writes a valid ``parity-witness`` beside the row under fit.png's placement
rules. Fixtures only: no cell script is run (the scripts are read as text / AST).
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
import catalogue_lifecycle as lc  # noqa: E402
import parity_witness  # noqa: E402

import _production_config as prod  # noqa: E402
import _profile_cli as pc  # noqa: E402

INSTRUMENTS = ("ao", "euclid", "hst", "jwst")
#: The imaging cell scripts the phase-2a campaign runs, by the variant whose size they build.
SIZED = {
    "scripts/imaging/delaunay/likelihood_runtime.py": "jax",
    "scripts/imaging/delaunay/likelihood_breakdown.py": "jax",
    "scripts/imaging/delaunay_nn/likelihood_runtime.py": "jax",
    "scripts/imaging/delaunay_nn/likelihood_breakdown.py": "jax",
    "scripts/imaging/rectangular/likelihood_runtime.py": "jax",
    "scripts/imaging/rectangular/likelihood_breakdown.py": "jax",
    "scripts/imaging/delaunay/likelihood_runtime_numba.py": "numba",
    "scripts/imaging/delaunay/likelihood_breakdown_numba.py": "numba",
    "scripts/imaging/rectangular/likelihood_runtime_numba.py": "numba",
    "scripts/imaging/rectangular/likelihood_breakdown_numba.py": "numba",
}
RUNTIMES = [
    "scripts/imaging/mge/likelihood_runtime.py",
    *(s for s in SIZED if "likelihood_runtime" in s),
]


# ------------------------------------------------------------------------- the CLI flag


def test_standard_size_counts_per_variant_and_one_copy_with_the_catalogue():
    cli = pc.parse_profile_cli(argv=["--standard-size"])
    assert cli.standard_size and cli.source_pixels is None
    assert pc.requested_source_pixels(cli, "delaunay") == 1250
    assert pc.requested_source_pixels(cli, "delaunay_nn") == 1250
    assert pc.requested_source_pixels(cli, "rectangular") == 1225
    assert pc.rectangular_side(1225, 39) == 35 and pc.rectangular_side(None, 39) == 39
    assert pc.STANDARD_SOURCE_PIXELS == lc.STANDARD_SOURCE_PIXELS
    plain = pc.parse_profile_cli(argv=[])
    assert not plain.standard_size and pc.requested_source_pixels(plain, "delaunay") is None
    scan = pc.parse_profile_cli(argv=["--source-pixels", "3000"])
    assert pc.requested_source_pixels(scan, "delaunay") == 3000


@pytest.mark.parametrize(
    "flag",
    [
        ["--source-pixels", "1250"],
        ["--rect-mesh", "rtu"],
        ["--rect-mesh=bilinear"],
        ["--over-sampling", "production"],
        ["--regularization", "constant_split"],
        ["--variant", "legacy"],
    ],
)
def test_standard_size_refuses_every_flag_it_would_silently_replace(flag, capsys):
    with pytest.raises(SystemExit):
        pc.parse_profile_cli(argv=["--standard-size", *flag])
    assert "would silently replace" in capsys.readouterr().err
    pc.parse_profile_cli(argv=flag)  # each is fine on its own
    assert set(pc.STANDARD_SIZE_EXCLUSIVE) == {
        "--source-pixels",
        "--rect-mesh",
        "--over-sampling",
        "--regularization",
        "--variant",
    }


def test_production_data_records_the_truth_lens_light_stand_in():
    preset = prod.standard_size_preset("pixelization_numba", "hst")
    assert preset.as_json()["production_data"] == {"enabled": True, "lens_light_source": "truth"}


def test_standard_size_rows_keep_the_canonical_name(tmp_path):
    std = pc.parse_profile_cli(argv=["--standard-size", "--output-dir", str(tmp_path)])
    path, _ = pc.resolve_output_paths(std, tmp_path, "delaunay_likelihood_summary_hst_v1")
    assert path.name == "delaunay_likelihood_summary_hst_v1.json"
    scan = pc.parse_profile_cli(argv=["--source-pixels", "1250", "--output-dir", str(tmp_path)])
    path, _ = pc.resolve_output_paths(scan, tmp_path, "delaunay_likelihood_summary_hst_v1")
    assert path.name == "delaunay_likelihood_summary_hst_v1_n1250.json"


# ---------------------------------------------------------------------- numba presets


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_numba_presets_at_the_standard_size_for_every_instrument(instrument):
    delaunay = prod.standard_size_preset("delaunay_numba", instrument)
    assert (delaunay.hilbert_pixels, delaunay.edge_pixels, delaunay.mesh_vertices) == (
        1250,
        30,
        1280,
    )
    rect = prod.standard_size_preset("pixelization_numba", instrument)
    assert rect.rect_pixels_yx == 35 and rect.as_json()["mesh"]["source_pixels"] == 1225
    if instrument in ("jwst", "ao"):  # no production reference: the hst preset, recorded
        for preset in (delaunay, rect):
            assert preset.name.endswith(f"_standard_{instrument}")
            assert "borrows the hst preset" in preset.size_override
            assert preset.as_json()["size_override"] == preset.size_override
        with pytest.raises(KeyError):
            prod.preset_for("delaunay_numba", instrument)


def test_every_standard_preset_is_production_data_and_retires_pin_and_witness():
    preset = prod.standard_size_preset("delaunay_numba", "hst")  # already 1250 + 30 in size
    assert preset.mesh_vertices == 1280 and preset.production_data
    assert "production data" in preset.size_override
    assert not prod.preset_for("delaunay_numba", "hst").production_data  # legacy untouched
    assert "production_data" not in prod.preset_for("delaunay_numba", "hst").as_json()
    resized = prod.standard_size_preset("delaunay_numba", "euclid")
    assert resized.size_override and "500 -> 1250" in resized.size_override
    witness = prod.witness_for(resized, 1.0, "euclid", "hpc_ral_cpu_fp64")
    assert witness["verdict"] == "not_applicable"
    base = prod.preset_for("delaunay_numba", "hst")
    assert prod.witness_for(base, 1.0, "hst", "local_laptop_fp64")["verdict"] == "INCONCLUSIVE"


def test_a_delaunay_nn_numba_cell_is_never_judged_against_the_delaunay_production_band():
    preset = prod.preset_for("delaunay_numba", "hst")  # the recorded preset: a band applies
    assert prod.witness_for(preset, 0.6, "hst", "hpc_ral_cpu_fp64")["verdict"] == "PASS"
    nn = prod.witness_for(preset, 0.6, "hst", "hpc_ral_cpu_fp64", mesh_class="delaunay_nn")
    assert nn["verdict"] == "not_applicable" and "delaunay_nn" in nn["reason"]
    for script in ("likelihood_runtime_numba.py", "likelihood_breakdown_numba.py"):
        text = (ROOT / "scripts/imaging/delaunay" / script).read_text()
        assert "_cli.config_name, mesh_class=MESH_CLASS" in text


# ------------------------------------------------------------------- script wiring


@pytest.mark.parametrize("script", sorted(SIZED))
def test_every_campaign_script_honours_standard_size_and_records_the_request(script):
    text = (ROOT / script).read_text()
    if SIZED[script] == "numba":
        assert "standard_size_preset(" in text and "witness_for(" in text
        assert "preset.size_override is not None" in text  # the pin is skipped when resized
    else:
        assert "requested_source_pixels(_cli" in text
        assert "_cli.source_pixels" not in text  # one route for --source-pixels / --standard-size
    assert "source_pixels_requested" in text and '"standard_size": bool(_cli.standard_size)' in text


def test_the_delaunay_numba_scripts_build_delaunay_nn_on_request():
    for script in ("likelihood_runtime_numba.py", "likelihood_breakdown_numba.py"):
        text = (ROOT / "scripts/imaging/delaunay" / script).read_text()
        assert '"--mesh-class", choices=("delaunay", "delaunay_nn")' in text
        assert "al.mesh.DelaunayNN if MESH_CLASS" in text
        assert "needs --output-path" in text  # the legacy name would place it under delaunay


def _calls(tree, name):
    return [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name
    ]


@pytest.mark.parametrize("script", RUNTIMES)
def test_every_campaign_runtime_writes_parity_beside_its_row_before_fit_png(script):
    tree = ast.parse((ROOT / script).read_text())
    [call] = _calls(tree, "write_parity_json")
    assert ast.unparse(call.args[0]) == "dict_path.parent"
    kwargs = {k.arg for k in call.keywords}
    assert {"log_likelihood", "reference_log_likelihood", "use_mixed_precision"} <= kwargs
    assert {"reference_source", "autolens_version", "source_pixels_requested"} <= kwargs
    statement = next(n for n in tree.body if any(c is call for c in ast.walk(n)))
    [render] = _calls(tree, "render_fit_png")
    writes = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "write_text"
        and ast.unparse(n.func.value) == "dict_path"
    ]
    assert max(writes) < statement.lineno < render.lineno  # after the row and every timing


def test_breakdowns_leave_the_cells_witness_to_the_runtime():
    for path in sorted((ROOT / "scripts/imaging").rglob("likelihood_breakdown*.py")):
        assert "write_parity_json" not in path.read_text(), path


# ----------------------------------------------------------------------- size rule


def test_size_rule_matches_on_a_standard_request_and_keeps_old_rows():
    standard = {"configuration": {"source_pixels": 1280, "source_pixels_requested": 1250}}
    assert lc.size_pixels("delaunay", standard) == 1250
    assert lc.regime("delaunay", lc.size_pixels("delaunay", standard)) == "standard"
    # A pre-ruling rectangular scan that asked for 1250 built 35x35 and is standard by count.
    old_scan = {"configuration": {"mesh_shape": [35, 35], "source_pixels_requested": 1250}}
    assert lc.size_pixels("rectangular", old_scan) == 1225
    asked_1500 = {"configuration": {"source_pixels": 1521, "source_pixels_requested": 1500}}
    assert lc.regime("rectangular", lc.size_pixels("rectangular", asked_1500)) == (
        "source_pixels_1521"
    )
    no_request = {"configuration": {"delaunay_vertices": 1500}}
    assert lc.size_pixels("delaunay", no_request) == 1500
    witness = {"source_pixels": 1280, "source_pixels_requested": 1250}  # a parity.json
    assert lc.size_pixels("delaunay", witness) == 1250


# ------------------------------------------------------------------- parity writer


@pytest.fixture
def no_smoke(monkeypatch):
    monkeypatch.delenv("AUTOLENS_PROFILING_SMOKE", raising=False)


@pytest.fixture
def cell(tmp_path):
    cell = tmp_path / "results/imaging/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu"
    cell.mkdir(parents=True)
    (cell / "runtime.json").write_text("{}")
    return cell


def witness_kwargs(**extra):
    kwargs = dict(
        log_likelihood=28128.9182887,
        reference_log_likelihood=28128.9182889,
        use_mixed_precision=False,
        reference_source="fixture: eager NumPy reference",
        autolens_version="2026.10.23.1",
        source_pixels=1280,
        source_pixels_requested=1250,
    )
    kwargs.update(extra)
    return kwargs


def test_writes_a_valid_match_witness_beside_the_row(cell, no_smoke):
    path = pc.write_parity_json(cell, **witness_kwargs())
    assert path == cell / "parity.json"
    doc = json.loads(path.read_text())
    assert parity_witness.validate(doc) == []
    assert doc["status"] == "match"
    assert doc["tolerance"] == pytest.approx(1e-4 * 28128.9182889)
    assert (doc["source_pixels"], doc["source_pixels_requested"]) == (1280, 1250)
    assert sorted(p.name for p in cell.iterdir()) == ["parity.json", "runtime.json"]
    assert parity_witness.main(["--root", str(cell.parents[5]), "--check"]) == 0


def test_status_follows_the_tolerance_and_mixed_precision_widens_it(cell, no_smoke):
    off = witness_kwargs(log_likelihood=1000.2, reference_log_likelihood=1000.0)
    assert json.loads(pc.write_parity_json(cell, **off).read_text())["status"] == "diverged"
    mixed = {**off, "use_mixed_precision": True}
    doc = json.loads(pc.write_parity_json(cell, **mixed).read_text())
    assert doc["status"] == "match" and doc["tolerance"] == pytest.approx(1.0)
    assert parity_witness.validate(doc) == []


def test_non_finite_values_are_not_assessed(cell, no_smoke):
    doc = json.loads(
        pc.write_parity_json(cell, **witness_kwargs(log_likelihood=float("nan"))).read_text()
    )
    assert doc["status"] == "not_assessed" and doc["reason"]
    assert parity_witness.validate(doc) == []


def test_a_row_rewritten_without_a_witness_loses_the_stale_one(cell, no_smoke):
    pc.write_parity_json(cell, **witness_kwargs())
    pc.remove_parity_json(cell)
    assert not (cell / "parity.json").exists()


def test_smoke_and_non_cells_touch_nothing(cell, tmp_path, monkeypatch, no_smoke):
    flat = tmp_path / "results/misc/likelihood_runtime/imaging"
    flat.mkdir(parents=True)
    (flat / "parity.json").write_text("other")
    assert pc.write_parity_json(flat, **witness_kwargs()) is None
    assert (flat / "parity.json").read_text() == "other"
    (cell / "parity.json").write_text("old")
    monkeypatch.setenv("AUTOLENS_PROFILING_SMOKE", "1")
    assert pc.write_parity_json(cell, **witness_kwargs()) is None
    assert (cell / "parity.json").read_text() == "old"


def test_a_scratch_tree_laid_out_like_results_is_a_cell(tmp_path):
    scratch = tmp_path / "proof/results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop"
    assert pc.fit_png_cell(scratch)
    assert not pc.fit_png_cell(tmp_path / "proof")
    assert not pc.fit_png_cell(tmp_path / "proof/results/imaging/rectangular/hst")
