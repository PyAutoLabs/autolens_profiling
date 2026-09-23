"""Static checks for the phase-3 PSF-cube A100 submit (#295).

**No imaging, no JAX, no GPU** — these read the submit script and the cell as text.

These legs run the *same* cell as the phase-1 trace family
(``scripts/imaging/likelihood_breakdown/fixed_light_trace.py``) in its
``--psf-candidate`` mode, so they are a sixth ``fixed_light`` submit family with
their own file-name prefix (``..._fixed_light_psf_cube_*``), excluded from
``test_fixed_light_cell.py``'s glob for the reason that file states. This file is
the PSF family's half of that bargain.

What a wrong submit would look like, and what catches it:

- the candidate table and the ``--array`` range disagree, so a task runs with an
  empty candidate — ``test__array_range_matches_the_candidates``;
- a task names a candidate the cell or the injection module does not know —
  ``test__every_candidate_is_one_the_cell_and_the_module_accept``;
- the submit runs against a checkout from **before** #295, which would reject
  the flag only after the GPU was allocated — the stale-checkout guard;
- the array loses its control, so nothing is matched — ``test__the_array_is_the_whole_registry``;
- a failing gate eats the footer, as phase 2's ERR trap did.
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

#: Shared with phase 1 and 2; the ``_psf_<candidate>`` suffix keeps the files disjoint.
CONFIG_NAME = "hpc_a100_fp64_fixed_light_trace"


def _submits() -> list[_Path]:
    return sorted(BATCH_GPU.glob("submit_breakdown_imaging_fixed_light_psf_cube_*"))


def _arm_table(text: str, name: str) -> list[str]:
    """The values of a ``NAME=(a b c)`` bash array."""
    match = re.search(rf"^{name}=\((.*?)\)", text, re.M)
    return match.group(1).split() if match else []


def _executable(text: str) -> str:
    """The submit with whole-line comments blanked, as check_submits reads it."""
    return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in text.splitlines())


def _registry() -> list[str]:
    from likelihood_breakdown.psf_cube_injection import CANDIDATES

    return list(CANDIDATES)


@pytest.fixture(scope="module")
def submits() -> list[_Path]:
    found = _submits()
    assert found, "no submit_breakdown_imaging_fixed_light_psf_cube_* submits found"
    return found


def test__the_psf_family_is_excluded_from_the_phase_0_glob():
    other = (ROOT / "scripts" / "misc" / "test" / "test_fixed_light_cell.py").read_text()
    assert '"_fixed_light_psf_cube_" not in p.name' in other


def test__the_psf_family_is_not_swallowed_by_the_trace_family_glob(submits):
    for path in submits:
        assert "_fixed_light_trace_" not in path.name
        assert "_fixed_light_vmap_" not in path.name


def test__the_cell_declares_every_phase_3_flag():
    text = CELL.read_text()
    for flag in ("--psf-candidate", "--pin-draws", "--draw-seed"):
        assert flag in text, f"cell does not declare {flag}"


def test__the_cell_records_the_phase_3_flags_and_suffixes_the_filename():
    text = CELL.read_text()
    for key in ("psf_candidate", "pin_draws", "draw_seed"):
        assert f'"{key}"' in text
    assert '_cell_name = f"{_cell_name}_psf_{PSF_CANDIDATE}"' in text


def test__the_cell_enforces_the_gate_after_writing_the_evidence():
    text = CELL.read_text()
    write = text.index("dict_path.write_text(")
    gate = text.index("failed the phase-3 gate")
    assert write < gate, "the gate must raise only after the JSON is written"
    assert "EQUIVALENCE_RTOL = 1.0e-9" in text
    assert "abs(_recon) <= 5.0" in text
    assert "_unjoined > 0.0" in text


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
        assert name in _registry(), f"{path.name}: {name!r} is not in psf_cube_injection"
        assert f'"{name}"' in cell, f"{path.name}: the cell's choices do not offer {name!r}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__the_array_is_the_whole_registry(path):
    """Control first, every lever, both diagnostics — nothing unmatched."""
    assert _arm_table(path.read_text(), "CANDIDATES") == _registry()


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__array_range_matches_the_candidates(path):
    text = path.read_text()
    n = len(_arm_table(text, "CANDIDATES"))
    array = re.search(r"^#SBATCH --array=(\d+)-(\d+)", text, re.M)
    assert array, f"{path.name}: no `#SBATCH --array=` line"
    assert int(array.group(1)) == 0
    assert int(array.group(2)) == n - 1, f"--array=0-{array.group(2)} but {n} candidates"
    assert "FATAL: no PSF candidate for array task" in text


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_passes_the_phase_3_flags(path):
    executable = _executable(path.read_text())
    for flag in (
        "--psf-candidate $CAND",
        "--pin-draws 8",
        "--draw-seed 0",
        "--routes b,d",
        "--mesh $MESH",
        "--border-relocator $BORDER",
        '--trace-dir "$TRACE_DIR"',
    ):
        assert flag in executable, f"{path.name}: does not pass {flag}"
    assert "--vmap-batch" not in executable, "the PSF mode rejects --vmap-batch"
    assert "--use-mixed-precision" not in executable


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_guards_against_a_checkout_without_the_flag(path):
    executable = _executable(path.read_text())
    assert 'grep -q -- "--psf-candidate"' in executable
    assert "This checkout predates #295" in executable


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
