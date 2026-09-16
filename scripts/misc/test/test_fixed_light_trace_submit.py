"""Static checks for the ``fixed_light_trace`` A100 submit family (#268).

**No imaging, no JAX, no GPU** — these read the submit script and the cell as text.

The trace legs are the *third* ``fixed_light`` family. Phase 0's cell
(`test_fixed_light_cell.py`) and the phase-1 library-path legs
(`test_fixed_light_library.py`) each own their own submits, and both files
exclude the others from their glob for a stated reason: the config names nest as
substrings (``hpc_a100_fp64_fixed_light`` ⊂
``hpc_a100_fp64_fixed_light_library`` ⊂ ``hpc_a100_fp64_fixed_light_trace``), so
an un-excluded family passes the wrong file's assertions **vacuously**. This file
is the trace family's half of that bargain — without it, excluding the trace
submits from `test_fixed_light_cell.py` would leave them checked by nothing.

What a wrong submit would look like, and what catches it:

- an arm table and an ``--array`` range that disagree, so a task silently runs
  with an empty mesh (the submit's own ``FATAL`` guard would catch it at run
  time, on the cluster, after queueing) — ``test__array_range_matches_the_arms``;
- a mesh or border mode the cell's argparse would reject, which
  ``parse_known_args`` does **not** reject — it ignores unknown flags, so the leg
  would run and quietly produce a different table;
- the pass budget left at phase 0's certifying budget (2 on Delaunay) instead of
  phase 3's production budget (7) — the defect the cell's ``PHASE3_SAFE_BUDGET``
  exists to prevent, and the reason no ``--pass-budget`` is passed at all.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_trace_submit.py
"""

from __future__ import annotations

import re
import subprocess
import sys as _sys
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
BATCH_GPU = ROOT / "hpc" / "batch_gpu"
CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_trace.py"

_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

#: The meshes and border modes the cell's argparse accepts.
MESHES = ("rectangular", "delaunay", "delaunay_nn")
BORDERS = ("cell", "library", "off")

#: The config name these legs write under. Outside ``build_readme.py``'s
#: ``CONFIG_TAGGED_RE``, so nothing here reaches the dashboard.
CONFIG_NAME = "hpc_a100_fp64_fixed_light_trace"


def _submits() -> list[_Path]:
    return sorted(BATCH_GPU.glob("submit_breakdown_imaging_fixed_light_trace_*"))


def _arm_table(text: str, name: str) -> list[str]:
    """The values of a ``NAME=(a b c)`` bash array."""
    match = re.search(rf"^{name}=\((.*?)\)", text, re.M)
    return match.group(1).split() if match else []


@pytest.fixture(scope="module")
def submits() -> list[_Path]:
    found = _submits()
    assert found, "no submit_breakdown_imaging_fixed_light_trace_* submits found"
    return found


def test__the_trace_family_is_excluded_from_the_other_families_globs():
    """Or its submits pass the phase-0 file's assertions by substring luck."""
    other = (ROOT / "scripts" / "misc" / "test" / "test_fixed_light_cell.py").read_text()
    assert '"_fixed_light_trace_" not in p.name' in other, (
        "test_fixed_light_cell.py no longer excludes the trace family; its "
        "--config-name assertion would pass on a trace leg only because "
        "hpc_a100_fp64_fixed_light_trace contains hpc_a100_fp64_fixed_light"
    )


def test__the_cell_exists_and_declares_its_flags():
    assert CELL.is_file(), f"{CELL} is missing"
    text = CELL.read_text()
    for flag in ("--mesh", "--border-relocator", "--trace-calls", "--routes", "--command-buffers"):
        assert flag in text, f"cell does not declare {flag}"


def test__the_cell_runs_the_production_pass_budget_not_the_certifying_one():
    """7 on Delaunay (phase 3, zero-fallback), never phase 0's certifying 2."""
    text = CELL.read_text()
    assert "PHASE3_SAFE_BUDGET" in text
    assert re.search(r'PHASE3_SAFE_BUDGET\s*=\s*\{[^}]*"delaunay":\s*7', text), (
        "the cell's default Delaunay budget is not 7 — phase 0's certifying budget of 2 "
        "falls back on 67.5 % of a graded draw set and is not a production cost"
    )


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
def test__every_arm_names_a_mesh_and_a_border_mode_the_cell_accepts(path):
    """``parse_known_args`` IGNORES an unknown flag value, so argparse will not catch this."""
    text = path.read_text()
    meshes = _arm_table(text, "MESHES")
    borders = _arm_table(text, "BORDERS")
    assert meshes, f"{path.name}: no MESHES=(...) arm table"
    assert borders, f"{path.name}: no BORDERS=(...) arm table"
    for mesh in meshes:
        assert mesh in MESHES, f"{path.name}: mesh {mesh!r} is not one of {MESHES}"
    for border in borders:
        assert border in BORDERS, f"{path.name}: border {border!r} is not one of {BORDERS}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__array_range_matches_the_arms(path):
    """A task with no arm would run with an empty mesh — on the cluster, after queueing."""
    text = path.read_text()
    meshes = _arm_table(text, "MESHES")
    borders = _arm_table(text, "BORDERS")
    assert len(meshes) == len(borders), (
        f"{path.name}: {len(meshes)} meshes but {len(borders)} border modes — "
        f"the arm tables are indexed by the same SLURM_ARRAY_TASK_ID"
    )
    array = re.search(r"^#SBATCH --array=(\d+)-(\d+)", text, re.M)
    assert array, f"{path.name}: no `#SBATCH --array=` line"
    first, last = int(array.group(1)), int(array.group(2))
    assert first == 0, f"{path.name}: array starts at {first}, but the arm tables are 0-indexed"
    assert last == len(meshes) - 1, (
        f"{path.name}: --array=0-{last} but only {len(meshes)} arms are defined"
    )
    assert "FATAL: no (mesh, border) arm" in text, (
        f"{path.name}: no guard for a task index with no arm"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_writes_under_the_trace_config_name(path):
    text = path.read_text()
    assert f"--config-name {CONFIG_NAME}" in text, (
        f"{path.name}: legs must write under --config-name {CONFIG_NAME}"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_does_not_pass_flags_the_cell_would_silently_swallow(path):
    """``parse_known_args`` ignores unknown flags, so a stale flag is invisible.

    ``--pins`` belongs to the library-path cell, not this one; passing it here
    would look like it selected a pin mode and do nothing at all.
    """
    executable = "\n".join(
        "" if ln.lstrip().startswith("#") else ln for ln in path.read_text().splitlines()
    )
    for flag in ("--pins", "--vmap-batch", "--pass-budget", "--dataset"):
        assert flag not in executable, (
            f"{path.name}: passes {flag}, which this cell does not define — "
            f"parse_known_args would ignore it silently"
        )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_starts_from_a_fresh_compilation_cache_and_censuses_it(path):
    """A seeded XLA autotune cache silently changes kernels (F 4.8 vs 25.6 ms)."""
    text = path.read_text()
    assert "JAX_COMPILATION_CACHE_DIR=" in text
    assert 'rm -rf "$JAX_COMPILATION_CACHE_DIR"' in text, (
        f"{path.name}: the compilation cache is set but never wiped"
    )
    assert "AUTOTUNE_ENTRIES count=" in text, (
        f"{path.name}: no autotune-cache census — nothing would notice a seeded cache"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_records_the_library_revisions_it_ran_against(path):
    text = path.read_text()
    for repo in ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens"):
        assert repo in text, f"{path.name}: does not echo {repo}'s revision"
    assert "autoarray.__file__" in text, (
        f"{path.name}: does not echo autoarray.__file__ — a leg could import a different "
        f"checkout from the one whose revision it printed"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_carries_a_wall_basis_block_that_the_gate_accepts(path):
    """The block itself is validated by wall/check_submits.py; this is its presence."""
    from wall.check_submits import check_text, parse_basis_rows

    text = path.read_text()
    rows = parse_basis_rows(text)
    assert rows, f"{path.name}: no WALL-BASIS block"
    assert all(row.get("cell", "").count("/") == 2 for row in rows), (
        f"{path.name}: a WALL-BASIS row's `cell` is not <dataset>/<cell>/<instrument> — "
        f"prose inside the block that looks like `key: value` is parsed as row data"
    )
    assert not check_text(text), f"{path.name}: {check_text(text)}"
