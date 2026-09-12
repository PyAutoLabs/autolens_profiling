"""Static checks on the ``imaging/fixed_light`` cell and its A100 submits.

The cell itself cannot be imported here: it runs its profile at module level
(every ``likelihood_breakdown`` cell does), and importing it would build the
fiducial model and fit it. What *is* checkable without a GPU, and what has
actually broken in this repo before, is the wiring around it:

- every ``submit_*fixed_light*`` script is valid bash (``bash -n``) and runs a
  script that exists at the path it names (the ``_recon_split`` legs shipped a
  flag the checked-out cell did not have — ``parse_known_args`` swallowed it and
  the run silently produced an unsplit table);
- the submits pass a ``--mesh`` the cell accepts, and the ``--config-name`` the
  harvest expects;
- the launcher lists exactly the legs that exist on disk.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_cell.py
"""

from __future__ import annotations

import re
import subprocess
import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_misc_dir = str(ROOT / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)

import pytest  # noqa: E402

CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light.py"
BATCH_GPU = ROOT / "hpc" / "batch_gpu"
LAUNCHER = BATCH_GPU / "submit_fixed_light.sh"

#: The meshes the cell's ``--mesh`` choices accept.
MESHES = ("rectangular", "delaunay", "delaunay_nn")

#: The config name the A100 legs write under. Deliberately absent from the
#: dashboard config names ``build_readme.py`` surfaces.
CONFIG_NAME = "hpc_a100_fp64_fixed_light"


def _submits() -> list[_Path]:
    return sorted(BATCH_GPU.glob("submit_breakdown_imaging_fixed_light_*"))


def test__the_cell_exists_and_declares_its_flags():
    assert CELL.is_file(), f"{CELL} is missing"
    text = CELL.read_text()
    for flag in ("--mesh", "--pass-budget-max", "--no-library-row", "--source-pixels"):
        assert flag in text, f"cell does not declare {flag}"


def test__at_least_the_three_fiducial_submits_exist():
    names = {p.name for p in _submits()}
    for mesh_label in ("pixelization", "delaunay", "delaunay_nn"):
        expected = f"submit_breakdown_imaging_fixed_light_{mesh_label}_a100_hst_fp64"
        assert expected in names, f"missing submit {expected}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_is_valid_bash(path):
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{path.name}: bash -n failed\n{result.stderr}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_runs_a_script_that_exists(path):
    text = path.read_text()
    scripts = re.findall(r"^python3 -u (\S+)", text, re.M)
    assert scripts, f"{path.name}: no `python3 -u <script>` line"
    for script in scripts:
        assert (ROOT / script).is_file(), f"{path.name}: runs missing script {script}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_passes_a_mesh_the_cell_accepts(path):
    text = path.read_text()
    meshes = re.findall(r"--mesh\s+(\S+)", text)
    assert meshes, f"{path.name}: the cell requires --mesh and the submit passes none"
    for mesh in meshes:
        assert mesh in MESHES, f"{path.name}: --mesh {mesh} is not one of {MESHES}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_writes_under_the_fixed_light_config_name(path):
    text = path.read_text()
    assert f"--config-name {CONFIG_NAME}" in text, (
        f"{path.name}: legs must write under --config-name {CONFIG_NAME} so the harvest "
        f"finds them and the dashboard does not"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_wipes_its_own_compilation_cache(path):
    # A seeded XLA autotune cache silently changes which kernels a leg uses
    # (results/notes/xla_autotune_triton_gemm.md: F at 4.8 vs 25.6 ms on the
    # same code). Every leg must start from an empty one.
    text = path.read_text()
    assert "JAX_COMPILATION_CACHE_DIR" in text, f"{path.name}: no per-job compilation cache"
    assert 'rm -rf "$JAX_COMPILATION_CACHE_DIR"' in text, (
        f"{path.name}: compilation cache directory is not wiped before the run"
    )


def test__launcher_exists_and_is_valid_bash():
    assert LAUNCHER.is_file(), f"{LAUNCHER} is missing"
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"submit_fixed_light.sh: bash -n failed\n{result.stderr}"


def test__launcher_lists_only_legs_that_exist():
    text = LAUNCHER.read_text()
    legs = re.findall(r"^submit_breakdown_imaging_fixed_light_\S+$", text, re.M)
    assert legs, "submit_fixed_light.sh names no legs"
    for leg in legs:
        assert (BATCH_GPU / leg).is_file(), f"launcher names missing leg {leg}"


def test__launcher_covers_every_fiducial_leg():
    text = LAUNCHER.read_text()
    for mesh_label in ("pixelization", "delaunay", "delaunay_nn"):
        leg = f"submit_breakdown_imaging_fixed_light_{mesh_label}_a100_hst_fp64"
        assert leg in text, f"launcher does not submit {leg}"
