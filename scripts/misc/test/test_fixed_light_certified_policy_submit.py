"""Static checks for the phase-B ``fixed_light_certified_policy`` A100 submit (#300).

**No imaging, no JAX, no GPU** — these read the submit script and the cell as text.

The legs run the phase-2 batched cell
(``scripts/imaging/likelihood_breakdown/fixed_light_trace.py --vmap-batch``) in
its ``--solver-source library`` mode: the solver is chosen through
``al.Settings`` and nothing is monkeypatched. They are a fifth ``fixed_light``
submit family with their own file-name prefix, excluded from
``test_fixed_light_cell.py``'s glob for the reason that file states (the config
names nest as substrings). This file is the family's half of that bargain.

What a wrong submit would look like, and what catches it:

- the arm tables and ``--array`` disagree, so a task runs with an empty arm —
  ``test__array_range_matches_the_arms``;
- a row is missing, so the policy verdict compares an incomplete set —
  ``test__the_array_is_the_pre_registered_task_table``;
- the submit runs a harness-only checkout, which would reject or swallow the
  library flags — ``test__submit_guards_against_a_checkout_without_the_flags``;
- the gate is not written down before submission —
  ``test__the_gate_is_pre_registered_in_the_submit_and_the_cell``.
"""

from __future__ import annotations

import itertools
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
SUBMIT = (
    ROOT
    / "hpc"
    / "batch_gpu"
    / "submit_breakdown_imaging_fixed_light_certified_policy_a100_hst_fp64"
)
CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_trace.py"

_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

CONFIG_NAME = "hpc_a100_fp64_fixed_light_trace"
TABLES = ("MESHES", "BATCHES", "LANESETS", "SOLVERS", "FALLBACKS")


def _arm_table(text: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\((.*?)\)", text, re.M)
    return match.group(1).split() if match else []


def _executable(text: str) -> str:
    return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in text.splitlines())


def _arms() -> list[tuple[str, ...]]:
    text = SUBMIT.read_text()
    return list(zip(*(_arm_table(text, name) for name in TABLES)))


def test__submit_is_valid_executable_bash():
    result = subprocess.run(["bash", "-n", str(SUBMIT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert SUBMIT.stat().st_mode & 0o111, "the submit is not executable"


def test__the_family_is_excluded_from_the_phase_0_glob():
    other = (ROOT / "scripts" / "misc" / "test" / "test_fixed_light_cell.py").read_text()
    assert '"_fixed_light_certified_policy_" not in p.name' in other


def test__array_range_matches_the_arms():
    text = SUBMIT.read_text()
    lengths = {name: len(_arm_table(text, name)) for name in TABLES}
    assert len(set(lengths.values())) == 1, lengths
    array = re.search(r"^#SBATCH --array=(\d+)-(\d+)", text, re.M)
    assert array and int(array.group(1)) == 0
    assert int(array.group(2)) == lengths["MESHES"] - 1
    assert "FATAL: no (mesh, batch, lanes, solver, fallback) arm" in text


def test__the_array_is_the_pre_registered_task_table():
    """18 distinct rows (mesh x B x row) then the two identical-lane controls."""
    rows = [("pdip", "on"), ("certified", "on"), ("certified", "off")]
    expected = [
        (mesh, str(batch), "distinct", solver, fallback)
        for mesh, batch, (solver, fallback) in itertools.product(
            ("delaunay", "rectangular"), (4, 8, 16), rows
        )
    ]
    expected += [
        ("delaunay", "16", "identical", "certified", "on"),
        ("rectangular", "16", "identical", "certified", "on"),
    ]
    assert _arms() == expected


def test__every_arm_is_a_setting_the_cell_accepts():
    for mesh, batch, lanes, solver, fallback in _arms():
        assert mesh in ("delaunay", "rectangular")
        assert batch.isdigit() and int(batch) >= 1
        assert lanes in ("distinct", "identical")
        assert solver in ("pdip", "certified")
        assert fallback in ("on", "off")


def test__submit_runs_the_library_mode_matched_pair():
    executable = _executable(SUBMIT.read_text())
    for flag in ("--solver-source library", "--solver $SOLVER", "--arms both", "--draw-seed 0"):
        assert flag in executable, f"the submit does not pass {flag}"
    assert f"--config-name {CONFIG_NAME}" in executable
    assert "--certified-budget" not in executable, (
        "the rows must run the library's resolved budget, the value production takes"
    )


def test__submit_guards_against_a_checkout_without_the_flags():
    executable = _executable(SUBMIT.read_text())
    assert 'grep -q -- "--solver-source"' in executable
    assert "predates #300" in executable
    assert "11b93476" in executable, "no guard that the mirror carries PyAutoArray#567"


def test__submit_preserves_footer_when_the_numerical_gate_fails():
    executable = _executable(SUBMIT.read_text())
    python_pos = executable.index("python3 -u")
    census_pos = executable.index("AUTOTUNE_ENTRIES count=")
    assert "SAVED_ERR_TRAP=$(trap -p ERR)" in executable[:python_pos]
    assert "trap - ERR" in executable[:python_pos]
    assert 'eval "$SAVED_ERR_TRAP"' in executable[python_pos:census_pos]
    assert "CELL_EXIT=$CELL_RC" in executable[census_pos:]


def test__submit_starts_from_a_fresh_compilation_cache_and_censuses_it():
    text = SUBMIT.read_text()
    assert 'rm -rf "$JAX_COMPILATION_CACHE_DIR"' in text
    assert 'rm -rf "$TRACE_DIR"' in text
    assert "SLURM_ARRAY_TASK_ID" in text.split("JAX_COMPILATION_CACHE_DIR=", 1)[1].splitlines()[0]


def test__submit_records_the_library_revisions_it_ran_against():
    text = SUBMIT.read_text()
    for repo in ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens"):
        assert repo in text
    assert "autoarray.__file__" in text


def test__the_gate_is_pre_registered_in_the_submit_and_the_cell():
    submit = SUBMIT.read_text()
    cell = CELL.read_text()
    for text in (submit, cell):
        assert "PRE-REGISTERED" in text
        assert "hst_gpu_residue_phase3_psf_2026_09.md" in text
    assert "GATED at 1e-9 relative:  the jit(vmap) arm vs the jit(vmap) library PDIP" in submit
    assert "GATED at 1e-9 relative:  the scalar jit arm vs the scalar jit library PDIP" in submit
    assert '"rel_diff_scalar_vs_library_pdip_scalar"' in cell
    assert "_ll_library_pdip_scalar" in cell


def test__the_cell_declares_and_records_the_phase_b_flags():
    cell = CELL.read_text()
    for flag in ("--solver-source", "--solver", "--certified-budget"):
        assert f'"{flag}"' in cell
    for key in ("solver_source", "certified_fallback", "certified_pass_budget", "solver_used"):
        assert f'"{key}"' in cell
    assert "_lib{LIBRARY_SOLVER}_fb" in cell, "the filename does not carry the library row"


def test__submit_carries_a_wall_basis_block_that_the_gate_accepts():
    from wall.check_submits import check_text, parse_basis_rows

    text = SUBMIT.read_text()
    assert parse_basis_rows(text)
    assert not check_text(text), check_text(text)
