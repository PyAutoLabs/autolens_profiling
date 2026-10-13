"""Current-runtime compile opt-in, isolation and small CPU correctness checks."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts/misc"))

from jax_compile import current, probe, update_pins


def request(root, *extra):
    return probe.parse_args(
        [
            "--objective",
            current.OBJECTIVE,
            "--transforms",
            "jit",
            "--output",
            str(root / "output" / current.OUTPUT_NAMESPACE / "run.json"),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--objective", "legacy"],
        ["--dataset-class", "interferometer"],
        ["--model-type", "pixelization"],
        ["--instrument", "jwst"],
        ["--mixed-precision"],
        ["--transforms", "vag"],
        ["--transforms", "jit,vmap"],
        ["--n-batch", "32"],
        ["--batch-size", "8"],
    ],
)
def test_unsupported_requests_fail_before_setup(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(current, "from_dataset", lambda *a: pytest.fail("setup entered"))
    with pytest.raises(ValueError):
        current.build_workload(request(tmp_path, *extra), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_output_must_be_explicit_new_and_isolated(tmp_path):
    args = request(tmp_path)
    for output in (
        None,
        str(tmp_path / "scripts/misc/jax_compile/results/local_cpu/mge.json"),
        str(tmp_path / "results/new.json"),
    ):
        args.output = output
        with pytest.raises(ValueError):
            current.validate_request(args, tmp_path)
    args = request(tmp_path)
    output = current.validate_request(args, tmp_path)
    current.write_result(output, [{"compile_s": 1}], {"objective": current.OBJECTIVE})
    with pytest.raises(FileExistsError):
        current.validate_request(args, tmp_path)
    with pytest.raises(FileExistsError):
        current.write_result(output, [], {})
    assert json.loads(output.read_text())["comparable_to_legacy_pins"] is False


def test_symlink_cannot_redirect_output_into_legacy_tree(tmp_path):
    legacy = tmp_path / "scripts/misc/jax_compile/results"
    legacy.mkdir(parents=True)
    namespace = tmp_path / "output" / current.OUTPUT_NAMESPACE
    namespace.parent.mkdir()
    namespace.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError):
        current.validate_request(request(tmp_path), tmp_path)
    assert list(legacy.iterdir()) == []


def test_legacy_reader_does_not_ingest_new_envelope(tmp_path, monkeypatch):
    # Even accidental relocation of the envelope must not become a legacy pin.
    current.write_result(tmp_path / "local_cpu/mge.json", [{"compile_s": 1}], {})
    monkeypatch.setattr(update_pins, "RESULTS", tmp_path)
    assert update_pins.corpus() == []


def test_main_writes_isolated_result_with_split_costs(tmp_path, monkeypatch):
    args = request(tmp_path)
    params = {"parameter": object()}
    workload = SimpleNamespace(
        objective=lambda p: p, params=params, model=SimpleNamespace(total_free_parameters=8)
    )
    monkeypatch.setattr(probe, "_WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(probe, "parse_args", lambda: args)
    monkeypatch.setattr(probe, "hardware_label", lambda jax: "test_cpu")
    monkeypatch.setattr(current, "build_workload", lambda *a: workload)
    monkeypatch.setattr(current, "provenance", lambda *a: {"fixture": True})

    def measure(name, f, x0, *a, **kw):
        assert name == "jit" and x0 is params
        return {
            "transform": name,
            "trace_s": 1,
            "compile_s": 2,
            "first_s": 3,
            "steady_s": 4,
            "cache_state": "none",
        }

    monkeypatch.setattr(probe, "measure", measure)
    probe.main()
    doc = json.loads(Path(args.output).read_text())
    rec = doc["records"][0]
    assert [rec[k] for k in ("trace_s", "compile_s", "first_s", "steady_s")] == [1, 2, 3, 4]
    assert rec["objective"] == current.OBJECTIVE
    assert rec["input_representation"] == "registered-model-instance-pytree"
    assert not (tmp_path / "scripts").exists()


def test_imports_are_side_effect_free(tmp_path):
    code = f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts/misc")!r})
import cell_builders.imaging_mge
import jax_compile.current
import jax_compile.probe
assert not (set(('autolens', 'autofit', 'jax')) & set(sys.modules))
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, timeout=20)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "script,extra",
    [
        ("export_probe.py", ["--mode", "save"]),
        ("export_probe.py", ["--mode", "load"]),
        ("trace_profile.py", []),
        ("probe.py", []),
    ],
)
def test_legacy_clis_fail_without_producing_artifacts(tmp_path, script, extra):
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/misc/jax_compile" / script), *extra],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode != 0
    assert "legacy" in run.stderr.lower()
    assert list(tmp_path.iterdir()) == []


def test_tiny_cpu_objective_matches_eager_runtime_reference(tmp_path):
    # Bounded correctness exercise, never a stored profile or speed assertion.
    code = f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts/misc")!r})
import autolens as al
import jax
import numpy as np
from cell_builders.imaging_mge import prepare_dataset
from jax_compile.current import from_dataset, provenance
from types import SimpleNamespace
from pathlib import Path
import json
from jax_compile.probe import transformed_fn_and_arg

data = al.Array2D.no_mask(values=np.arange(121).reshape(11, 11) / 121 + 1, pixel_scales=0.1)
noise = al.Array2D.full(fill_value=0.1, shape_native=(11, 11), pixel_scales=0.1)
psf = al.Convolver.no_blur(pixel_scales=0.1)
dataset = prepare_dataset(al.Imaging(data=data, noise_map=noise, psf=psf))
w = from_dataset(dataset)
i = w.instance
assert len(i.galaxies.lens.bulge.profile_list) == len(i.galaxies.source.bulge.profile_list) == 20
assert i.galaxies.lens.mass.einstein_radius == 1.6
assert tuple(i.galaxies.lens.mass.centre) == (0., 0.)
assert i.fields.shear.gamma_1 == i.fields.shear.gamma_2 == 0.05
tracer = al.Tracer(galaxies=list(i.galaxies), fields=[i.fields])
reference = al.FitImaging(dataset=dataset, tracer=tracer,
                        settings=al.Settings(use_border_relocator=True), xp=np).figure_of_merit
fn, arg = transformed_fn_and_arg('jit', w.objective, w.params, 16, 4)
result = jax.jit(fn).lower(arg).compile()(arg)
assert np.isfinite(reference) and np.isfinite(result)
np.testing.assert_allclose(result, reference, rtol=1e-8, atol=1e-6)
assert jax.config.jax_enable_x64
assert all(str(x.dtype) == 'float64' for x in jax.tree_util.tree_leaves(w.params))
# Validate metadata serialization with explicitly synthetic input bytes.
for name in ('data.fits', 'psf.fits', 'noise_map.fits'):
    Path(name).write_bytes(b'test-only')
metadata = provenance(SimpleNamespace(dataset_path='.'), Path('.'), w)
assert metadata['workload']['model_configuration']
assert metadata['workload']['precision'] == 'fp64'
json.dumps(metadata, allow_nan=False)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        timeout=60,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"},
    )
    assert not (tmp_path / "results").exists()
