"""The per-cell ``fit.png`` (likelihood-lifecycle 1B): ``_profile_cli.render_fit_png`` and its wiring.

``render_fit_png`` draws a cell's eager fit with the plotter the autolens_visualization producer
uses for that dataset type and keeps only ``fit.png`` beside the placed result JSON. These tests
render one tiny simulated imaging fit (no profiling), check the smoke / ``--no-fit-png`` switches
and that a figure failure never raises, and pin every wired cell script to call the helper
after its last result write with the placed row's folder.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

#: Every profiled cell script that renders a fit.png: the 13 runtime scripts plus the two
#: breakdown-only cluster cells.
WIRED = {
    "scripts/imaging/mge/likelihood_runtime.py": "imaging",
    "scripts/imaging/rectangular/likelihood_runtime.py": "imaging",
    "scripts/imaging/rectangular/likelihood_runtime_numba.py": "imaging",
    "scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py": "imaging",
    "scripts/imaging/delaunay/likelihood_runtime.py": "imaging",
    "scripts/imaging/delaunay/likelihood_runtime_numba.py": "imaging",
    "scripts/imaging/delaunay_nn/likelihood_runtime.py": "imaging",
    "scripts/interferometer/mge/likelihood_runtime.py": "interferometer",
    "scripts/interferometer/rectangular/likelihood_runtime.py": "interferometer",
    "scripts/interferometer/delaunay/likelihood_runtime.py": "interferometer",
    "scripts/datacube/delaunay/likelihood_runtime.py": "interferometer",
    "scripts/point_source_image/image_plane/likelihood_runtime.py": "point_source",
    "scripts/point_source_source/source_plane/likelihood_runtime.py": "point_source",
    "scripts/cluster/image_plane/likelihood_breakdown.py": "cluster",
    "scripts/cluster/source_plane/likelihood_breakdown.py": "cluster",
}


@pytest.fixture(scope="module")
def imaging_fit():
    """A tiny two-plane imaging fit (eager NumPy), simulated in memory."""
    import autolens as al

    grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)
    psf = al.Convolver.from_gaussian(shape_native=(5, 5), sigma=0.1, pixel_scales=0.1)
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.8)),
            al.Galaxy(
                redshift=1.0,
                bulge=al.lp.SersicCore(
                    centre=(0.05, 0.05), intensity=1.0, effective_radius=0.2, sersic_index=1.0
                ),
            ),
        ]
    )
    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise_to_data=True
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    mask = al.Mask2D.circular(shape_native=(40, 40), pixel_scales=0.1, radius=1.6)
    return al.FitImaging(dataset=dataset.apply_mask(mask=mask), tracer=tracer)


@pytest.fixture
def no_smoke(monkeypatch):
    monkeypatch.delenv("AUTOLENS_PROFILING_SMOKE", raising=False)


def test_renders_fit_png_beside_the_row_and_keeps_nothing_else(imaging_fit, tmp_path, no_smoke):
    cell = tmp_path / "results/imaging/mge/hst/jax_cpu_dense_fp64/local_laptop"
    cell.mkdir(parents=True)
    (cell / "runtime.json").write_text("{}")

    path = pc.render_fit_png(imaging_fit, cell, dataset_type="imaging")

    assert path == cell / "fit.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 10_000
    assert sorted(p.name for p in cell.iterdir()) == ["fit.png", "runtime.json"]


def test_skipped_under_smoke(imaging_fit, tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLENS_PROFILING_SMOKE", "1")
    assert pc.render_fit_png(imaging_fit, tmp_path, dataset_type="imaging") is None
    assert list(tmp_path.iterdir()) == []


def test_no_fit_png_flag_turns_it_off(imaging_fit, tmp_path, no_smoke):
    assert pc.parse_profile_cli(argv=[]).fit_png is True
    cli = pc.parse_profile_cli(argv=["--no-fit-png"])
    assert cli.fit_png is False
    assert pc.render_fit_png(imaging_fit, tmp_path, dataset_type="imaging", cli=cli) is None
    assert list(tmp_path.iterdir()) == []


def test_a_failed_figure_never_raises_or_leaves_debris(tmp_path, no_smoke, capsys):
    assert pc.render_fit_png(object(), tmp_path, dataset_type="weak") is None
    assert list(tmp_path.iterdir()) == []
    assert "[fit.png] not rendered" in capsys.readouterr().out


def _render_calls(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "render_fit_png"
    ]


@pytest.mark.parametrize("script", sorted(WIRED))
def test_every_profiled_cell_renders_after_its_last_write(script):
    """One call per cell, at module level, after the row is written, into the row's folder."""
    tree = ast.parse((ROOT / script).read_text())
    [call] = _render_calls(tree)
    kwargs = {k.arg: k.value for k in call.keywords}
    assert ast.literal_eval(kwargs["dataset_type"]) == WIRED[script]
    assert isinstance(kwargs["cli"], ast.Name) and kwargs["cli"].id == "_cli"
    assert ast.unparse(call.args[1]) == "dict_path.parent"
    statement = next(
        node for node in tree.body if any(n is call for n in ast.walk(node))
    )  # module level, never inside a timed function
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "write_text"
        and ast.unparse(node.func.value) == "dict_path"
    ]
    assert writes and statement.lineno > max(writes)
    assert statement is tree.body[-1]  # last statement: after every timed region


def test_unwired_scripts_carry_no_render_call():
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel not in WIRED and "/test/" not in rel:
            assert "render_fit_png(" not in path.read_text(), rel
