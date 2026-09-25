"""Static checks for the phase-4 log-det reuse A100 submit (#303).

**No imaging, no JAX, no GPU** — these read the submit script and the cell as text.

These legs run the *same* cell as the phase-1 trace family
(``scripts/imaging/likelihood_breakdown/fixed_light_trace.py``) in its
``--logdet-candidate`` mode, so they are a seventh ``fixed_light`` submit family
with their own file-name prefix (``..._fixed_light_logdet_reuse_*``), excluded
from ``test_fixed_light_cell.py``'s glob for the reason that file states. This
file is the log-det family's half of that bargain; it mirrors
``test_fixed_light_psf_cube_submit.py``.

What a wrong submit would look like, and what catches it:

- the (mesh, candidate) tables and the ``--array`` range disagree, so a task runs
  with an empty candidate — ``test__array_range_matches_the_tables``;
- a task names a candidate the cell or the injection module does not know —
  ``test__every_candidate_is_one_the_cell_and_the_module_accept``;
- a mesh loses its control, so its levers have nothing in-process to be compared
  with — ``test__each_mesh_runs_the_whole_registry``;
- the submit runs against a checkout from **before** #303 — the stale-checkout guard;
- a failing gate eats the footer — ``test__submit_preserves_footer_when_the_gate_fails``.
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

#: Shared with phases 1-3; the ``_logdet_<candidate>`` suffix keeps the files disjoint.
CONFIG_NAME = "hpc_a100_fp64_fixed_light_trace"
MESHES = ("delaunay", "rectangular")


def _submits() -> list[_Path]:
    return sorted(BATCH_GPU.glob("submit_breakdown_imaging_fixed_light_logdet_reuse_*"))


def _arm_table(text: str, name: str) -> list[str]:
    """The values of a ``NAME=(a b c)`` bash array."""
    match = re.search(rf"^{name}=\((.*?)\)", text, re.M)
    return match.group(1).split() if match else []


def _executable(text: str) -> str:
    """The submit with whole-line comments blanked, as check_submits reads it."""
    return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in text.splitlines())


def _registry() -> list[str]:
    from likelihood_breakdown.logdet_reuse_injection import CANDIDATES

    return list(CANDIDATES)


@pytest.fixture(scope="module")
def submits() -> list[_Path]:
    found = _submits()
    assert found, "no submit_breakdown_imaging_fixed_light_logdet_reuse_* submits found"
    return found


def test__the_logdet_family_is_excluded_from_the_phase_0_glob():
    other = (ROOT / "scripts" / "misc" / "test" / "test_fixed_light_cell.py").read_text()
    assert '"_fixed_light_logdet_reuse_" not in p.name' in other


def test__the_logdet_family_is_not_swallowed_by_another_family_glob(submits):
    for path in submits:
        for other in ("_fixed_light_trace_", "_fixed_light_vmap_", "_fixed_light_psf_cube_"):
            assert other not in path.name


def test__the_cell_declares_every_phase_4_flag():
    text = CELL.read_text()
    for flag in ("--logdet-candidate", "--pin-draws", "--draw-seed"):
        assert flag in text, f"cell does not declare {flag}"


def test__the_cell_records_the_phase_4_flags_and_suffixes_the_filename():
    text = CELL.read_text()
    for key in ("logdet_candidate", "pin_draws", "draw_seed", "certified_pass_budget"):
        assert f'"{key}"' in text
    assert '_cell_name = f"{_cell_name}_logdet_{LOGDET_CANDIDATE}"' in text


def test__the_cell_enforces_the_gate_after_writing_the_evidence():
    text = CELL.read_text()
    write = text.index("dict_path.write_text(")
    gate = text.index("failed the phase-4 gate")
    assert write < gate, "the gate must raise only after the JSON is written"
    assert "EQUIVALENCE_RTOL = 1.0e-9" in text
    assert "abs(_ld_recon) <= 5.0" in text
    assert "_ld_unjoined > 0.0" in text
    assert "logdet_reuse_injection.logdet_gate_rows(" in text


def test__the_cell_runs_the_library_solver_not_the_harness_one():
    text = CELL.read_text()
    assert 'positive_only_solver="certified"' in text
    assert 'certified_fallback="pdip"' in text
    assert "_settings_logdet" in text
    assert '!= ["certified"]' in text, "the cell must assert the library ran the certified solver"


def test__the_gate_and_lever_are_pre_registered_in_the_cell_docstring():
    text = CELL.read_text()
    doc = text[: text.index('"""', 3)]
    assert "The log-det mode (phase 4, autolens_profiling#303)" in doc
    assert "PRE-REGISTERED before any A100 submission" in doc
    assert ">= 0.5 ms" in doc


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_is_valid_bash(path):
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{path.name}: bash -n failed\n{result.stderr}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_runs_a_script_that_exists(path):
    scripts = re.findall(r"^python3 -u (\S+)", path.read_text(), re.M)
    assert scripts, f"{path.name}: no `python3 -u <script>` line"
    for script in scripts:
        assert (ROOT / script).is_file(), f"{path.name}: runs missing script {script}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__every_candidate_is_one_the_cell_and_the_module_accept(path):
    candidates = _arm_table(path.read_text(), "CANDIDATES")
    assert candidates, f"{path.name}: no CANDIDATES=(...) table"
    cell = CELL.read_text()
    for name in candidates:
        assert name in _registry(), f"{path.name}: {name!r} is not in logdet_reuse_injection"
        assert f'"{name}"' in cell, f"{path.name}: the cell's choices do not offer {name!r}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__each_mesh_runs_the_whole_registry(path):
    """Per mesh: control first, then every lever — each lever has an in-array control."""
    text = path.read_text()
    meshes = _arm_table(text, "MESHES")
    candidates = _arm_table(text, "CANDIDATES")
    assert len(meshes) == len(candidates)
    for mesh in MESHES:
        assert [c for m, c in zip(meshes, candidates) if m == mesh] == _registry()
    assert sorted(set(meshes)) == sorted(MESHES)


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__array_range_matches_the_tables(path):
    text = path.read_text()
    n = len(_arm_table(text, "CANDIDATES"))
    assert n == len(MESHES) * len(_registry())
    array = re.search(r"^#SBATCH --array=(\d+)-(\d+)", text, re.M)
    assert array, f"{path.name}: no `#SBATCH --array=` line"
    assert int(array.group(1)) == 0
    assert int(array.group(2)) == n - 1, f"--array=0-{array.group(2)} but {n} tasks"
    assert "FATAL: no log-det candidate for array task" in text


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_passes_the_phase_4_flags(path):
    executable = _executable(path.read_text())
    for flag in (
        "--logdet-candidate $CAND",
        "--pin-draws 8",
        "--draw-seed 0",
        "--routes b,d",
        "--mesh $MESH",
        "--border-relocator $BORDER",
        '--trace-dir "$TRACE_DIR"',
    ):
        assert flag in executable, f"{path.name}: does not pass {flag}"
    for rejected in ("--vmap-batch", "--psf-candidate", "--use-mixed-precision", "--solver "):
        assert rejected not in executable, f"{path.name}: the log-det mode rejects {rejected}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_guards_against_a_checkout_without_the_flag(path):
    executable = _executable(path.read_text())
    assert 'grep -q -- "--logdet-candidate"' in executable
    assert "This checkout predates #303" in executable


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_pre_registers_the_gate_and_the_lever(path):
    text = path.read_text()
    assert "PRE-REGISTERED before any A100 data" in text
    assert "1e-9 relative" in text and ">= 0.5 ms" in text


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_writes_under_the_trace_config_name(path):
    assert f"--config-name {CONFIG_NAME}" in path.read_text()


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_starts_from_a_fresh_compilation_cache_and_censuses_it(path):
    text = path.read_text()
    assert "JAX_COMPILATION_CACHE_DIR=" in text
    assert 'rm -rf "$JAX_COMPILATION_CACHE_DIR"' in text
    assert "AUTOTUNE_ENTRIES count=" in text


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_preserves_footer_when_the_gate_fails(path):
    executable = _executable(path.read_text())
    python_pos = executable.index("python3 -u")
    census_pos = executable.index("AUTOTUNE_ENTRIES count=")
    assert "SAVED_ERR_TRAP=$(trap -p ERR)" in executable[:python_pos]
    assert "trap - ERR" in executable[:python_pos]
    assert 'eval "$SAVED_ERR_TRAP"' in executable[python_pos:census_pos]
    assert "CELL_EXIT=$CELL_RC" in executable[census_pos:]


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_guards_the_shared_dataset(path):
    assert "dataset/imaging/hst/data.fits" in path.read_text()


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_records_the_library_revisions_it_ran_against(path):
    text = path.read_text()
    for repo in ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens"):
        assert repo in text
    assert "autoarray.__file__" in text


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_carries_a_wall_basis_block_that_the_gate_accepts(path):
    from wall.check_submits import check_text, parse_basis_rows

    text = path.read_text()
    rows = parse_basis_rows(text)
    assert rows, f"{path.name}: no WALL-BASIS block"
    assert [row.get("cell") for row in rows] == ["imaging/fixed_light_trace/hst"]
    assert check_text(text) == [], f"{path.name}: {check_text(text)}"
