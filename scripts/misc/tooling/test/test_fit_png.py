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

#: Every profiled cell script that renders a fit.png: the 13 runtime scripts, the two
#: breakdown-only cluster cells, and (likelihood-lifecycle 2c, #472) the point-source breakdowns
#: and solved-likelihood runtimes.
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
    "scripts/point_source_image/image_plane/likelihood_runtime_solved.py": "point_source",
    "scripts/point_source_source/source_plane/likelihood_runtime_solved.py": "point_source",
    "scripts/point_source_image/image_plane/likelihood_breakdown.py": "point_source",
    "scripts/point_source_source/source_plane/likelihood_breakdown.py": "point_source",
    "scripts/cluster/image_plane/likelihood_breakdown.py": "cluster",
    "scripts/cluster/source_plane/likelihood_breakdown.py": "cluster",
}
#: Likelihood-lifecycle 2b (#469): renders outside the one-call-per-cell module pattern. The
#: shared standard-size driver renders for the interferometer runtime / breakdown cells that call
#: it (after their last row write), and the matrix-free experiment renders beside its
#: experiments/ or scans/ row.
SHARED_RENDERERS = {
    "scripts/misc/likelihood_breakdown/interferometer_standard.py",
    "scripts/interferometer/rectangular/likelihood_runtime_matrixfree.py",
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


@pytest.fixture(scope="module")
def interferometer_fit():
    """A tiny two-plane interferometer fit (DFT, eager NumPy), simulated in memory."""
    import autolens as al
    import numpy as np

    grid = al.Grid2D.uniform(shape_native=(32, 32), pixel_scales=0.1)
    mask = al.Mask2D.circular(shape_native=(32, 32), pixel_scales=0.1, radius=1.4)
    uv = np.random.default_rng(1).normal(scale=1.0e5, size=(60, 2))
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
    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv,
        exposure_time=300.0,
        noise_sigma=0.1,
        transformer_class=al.TransformerDFT,
        noise_seed=1,
    )
    simulated = simulator.via_tracer_from(tracer=tracer, grid=grid)
    dataset = al.Interferometer(
        data=simulated.data,
        noise_map=simulated.noise_map,
        uv_wavelengths=uv,
        real_space_mask=mask,
        transformer_class=al.TransformerDFT,
    )
    return al.FitInterferometer(dataset=dataset, tracer=tracer)


def _point_dataset_and_tracer():
    """The tracked point_source/simple positions and true tracer (what the point cells use)."""
    import autolens as al

    path = ROOT / "dataset" / "point_source" / "simple"
    dataset = al.from_json(file_path=path / "point_dataset_positions_only.json")
    return dataset, al.from_json(file_path=path / "tracer.json")


@pytest.fixture(scope="module")
def point_fit():
    """An image-plane all-pairs fit with a PointSolver, as the point_source_image cell builds."""
    import autolens as al

    dataset, tracer = _point_dataset_and_tracer()
    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05),
        pixel_scale_precision=0.001,
    )
    return al.FitPointDataset(
        dataset=dataset,
        tracer=tracer,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePairAll,
    )


@pytest.fixture(scope="module")
def cluster_fits():
    """Two solver-free source-plane fits, as the cluster/source_plane cell wraps them."""
    import autolens as al

    dataset, tracer = _point_dataset_and_tracer()
    fit = al.FitPointDataset(
        dataset=dataset, tracer=tracer, solver=None, fit_positions_cls=al.FitPositionsSource
    )
    return [fit, fit]


@pytest.fixture
def no_smoke(monkeypatch):
    monkeypatch.delenv("AUTOLENS_PROFILING_SMOKE", raising=False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch repo root, so mirrored-tree cells can be made under ``tmp_path``."""
    monkeypatch.setattr(pc, "_REPO_ROOT", tmp_path)
    return tmp_path


def _cell(repo, *extra):
    cell = repo.joinpath("results/imaging/mge/hst/jax_cpu_dense_fp64/local_laptop", *extra)
    cell.mkdir(parents=True)
    (cell / "runtime.json").write_text("{}")
    return cell


def _assert_png(path):
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 10_000


def test_renders_fit_png_beside_the_row_and_keeps_nothing_else(imaging_fit, repo, no_smoke):
    cell = _cell(repo)
    (cell / "fit.png").write_bytes(b"stale")
    path = pc.render_fit_png(imaging_fit, cell, dataset_type="imaging")
    assert path == cell / "fit.png"
    _assert_png(path)
    assert sorted(p.name for p in cell.iterdir()) == ["fit.png", "runtime.json"]


@pytest.mark.parametrize(
    ("fixture", "dataset_type"),
    [
        ("interferometer_fit", "interferometer"),
        ("point_fit", "point_source"),
        ("cluster_fits", "cluster"),
    ],
)
def test_every_dataset_type_plotter_writes_fit_png(fixture, dataset_type, repo, no_smoke, request):
    cell = _cell(repo)
    path = pc.render_fit_png(request.getfixturevalue(fixture), cell, dataset_type=dataset_type)
    assert path == cell / "fit.png"
    _assert_png(path)


def test_experiment_arm_and_model_variant_folders_are_cells(imaging_fit, repo, no_smoke):
    for extra in (("experiments", "arm"), ("with_shear",)):
        cell = _cell(repo, *extra)
        assert pc.render_fit_png(imaging_fit, cell, dataset_type="imaging") == cell / "fit.png"


def test_smoke_touches_nothing(imaging_fit, repo, monkeypatch):
    """A smoke run writes no row, so a committed figure beside an old row must survive."""
    monkeypatch.setenv("AUTOLENS_PROFILING_SMOKE", "1")
    cell = _cell(repo)
    (cell / "fit.png").write_bytes(b"committed")
    assert pc.render_fit_png(imaging_fit, cell, dataset_type="imaging") is None
    assert (cell / "fit.png").read_bytes() == b"committed"


def test_no_fit_png_removes_the_previous_rows_figure(imaging_fit, repo, no_smoke):
    assert pc.parse_profile_cli(argv=[]).fit_png is True
    cli = pc.parse_profile_cli(argv=["--no-fit-png"])
    assert cli.fit_png is False
    cell = _cell(repo)
    (cell / "fit.png").write_bytes(b"stale")
    assert pc.render_fit_png(imaging_fit, cell, dataset_type="imaging", cli=cli) is None
    assert sorted(p.name for p in cell.iterdir()) == ["runtime.json"]


def test_a_failed_figure_removes_the_stale_one_and_never_raises(repo, no_smoke, capsys):
    cell = _cell(repo)
    (cell / "fit.png").write_bytes(b"stale")
    assert pc.render_fit_png(object(), cell, dataset_type="imaging") is None
    assert sorted(p.name for p in cell.iterdir()) == ["runtime.json"]
    assert "WARNING: [fit.png] NOT RENDERED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "folder",
    [
        "results/misc/likelihood_breakdown/cluster",  # flat --output-dir folder of HPC submits
        "results/runtime/imaging/mge/hst",  # legacy folder named verbatim
        "results/imaging/mge/hst",  # above the setup/host levels
    ],
)
def test_flat_or_legacy_folders_get_no_figure(imaging_fit, repo, no_smoke, folder):
    out = repo / folder
    out.mkdir(parents=True)
    (out / "fit.png").write_bytes(b"other")
    assert pc.render_fit_png(imaging_fit, out, dataset_type="imaging") is None
    assert (out / "fit.png").read_bytes() == b"other"


def test_outside_the_repo_gets_no_figure(imaging_fit, tmp_path, no_smoke):
    assert pc.render_fit_png(imaging_fit, tmp_path, dataset_type="imaging") is None
    assert list(tmp_path.iterdir()) == []


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
        if rel not in WIRED and rel not in SHARED_RENDERERS and "/test/" not in rel:
            assert "render_fit_png(" not in path.read_text(), rel
