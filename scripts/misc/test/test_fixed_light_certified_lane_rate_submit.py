"""Static checks for the phase-C1 ``fixed_light_certified_lane_rate`` A100 submit (#304).

**No imaging, no JAX, no GPU** — these read the submit script and the two cells as text.

The submit runs two cells: ``nautilus_batch_capture.py`` (tasks 0-3, the pix1 / pix2
captures on both meshes) and ``fixed_light_trace.py --lanes captured`` (tasks 4-37,
the rate and timing replays).
It is a sixth ``fixed_light`` submit family with its own file-name prefix, excluded
from ``test_fixed_light_cell.py``'s glob for the reason that file states; this file
is the family's half of that bargain, mirroring
``test_fixed_light_certified_policy_submit.py``.

What a wrong submit would look like, and what catches it:

- the arm tables and ``--array`` disagree — ``test__array_range_matches_the_arms``;
- a row of the verdict table is missing — ``test__the_array_is_the_pre_registered_task_table``;
- the replay starts before (or without) the capture —
  ``test__replay_refuses_a_missing_capture_and_documents_the_two_step``;
- the capture drifts from the production Nautilus settings —
  ``test__capture_runs_the_production_nautilus_settings``;
- the gate is not written down before submission, or is drawn tighter than 1e-9 —
  ``test__the_gate_is_pre_registered_and_no_tighter_than_1e_9``.
"""

from __future__ import annotations

import itertools
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
SUBMIT = (
    ROOT
    / "hpc"
    / "batch_gpu"
    / "submit_breakdown_imaging_fixed_light_certified_lane_rate_a100_hst_fp64"
)
CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_trace.py"
CAPTURE = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "nautilus_batch_capture.py"
PIPELINE_FULL_MODEL = (
    ROOT.parent / "euclid_strong_lens_modeling_pipeline" / "scripts" / "full_model.py"
)

_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

CONFIG_NAME = "hpc_a100_fp64_fixed_light_trace"
TABLES = ("MESHES", "STAGES", "PASSES", "BATCHES", "SOLVERS", "FALLBACKS")


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
    assert '"_fixed_light_certified_lane_rate_" not in p.name' in other


def test__array_range_matches_the_arms():
    text = SUBMIT.read_text()
    lengths = {name: len(_arm_table(text, name)) for name in TABLES}
    assert len(set(lengths.values())) == 1, lengths
    array = re.search(r"^#SBATCH --array=(\d+)-(\d+)", text, re.M)
    assert array and int(array.group(1)) == 0
    assert int(array.group(2)) == lengths["MESHES"] - 1
    assert "FATAL: no (mesh, stage, pass, batch, solver, fallback) arm" in text


def test__the_array_is_the_pre_registered_task_table():
    """4 captures, 4 rate passes, the pix1 B grid, then pix2 at B=20."""
    both = [
        (mesh, stage)
        for stage, mesh in itertools.product(("pix1", "pix2"), ("delaunay", "rectangular"))
    ]
    expected = [(mesh, stage, "capture", "20", "-", "-") for mesh, stage in both]
    expected += [(mesh, stage, "rate", "20", "certified", "off") for mesh, stage in both]
    rows = [("pdip", "on"), ("certified", "on"), ("certified", "off")]
    expected += [
        (mesh, "pix1", "time", str(batch), solver, fallback)
        for mesh, batch, (solver, fallback) in itertools.product(
            ("delaunay", "rectangular"), (16, 20, 50, 100), rows
        )
    ]
    expected += [
        (mesh, "pix2", "time", "20", solver, fallback)
        for mesh, (solver, fallback) in itertools.product(("delaunay", "rectangular"), rows)
    ]
    assert _arms() == expected


def test__replay_refuses_a_missing_capture_and_documents_the_two_step():
    text = SUBMIT.read_text()
    executable = _executable(text)
    assert 'if [ "$PASS" != "capture" ] && [ ! -f "$NPZ" ]; then' in executable
    assert "sbatch --parsable --array=0-3" in text
    assert "--array=4-37 --dependency=afterok:${CAP}" in text
    assert (
        'NPZ="$AP_ROOT/results/breakdown/imaging/nautilus_batches_${MESH}_${STAGE}.npz"'
        in executable
    )


def test__capture_runs_the_production_nautilus_settings():
    executable = _executable(SUBMIT.read_text())
    for flag in ("--stage $STAGE", "--n-batch 20", "--n-like-max 40000", "--capture-seconds"):
        assert flag in executable, f"the capture does not pass {flag}"
    assert "--n-live" not in executable, "n_live must come from the stage's production value"
    capture = CAPTURE.read_text()
    assert 'PRODUCTION_N_LIVE = {"pix1": 150, "pix2": 75}' in capture
    assert "PRODUCTION_N_BATCH = 20" in capture
    assert "use_jax_vmap=True" in capture
    if PIPELINE_FULL_MODEL.exists():
        # The production stage the capture mirrors still uses these settings.
        pipeline = PIPELINE_FULL_MODEL.read_text()
        source_pix_1 = pipeline.split("def source_pix_1(", 1)[1].split("\ndef ", 1)[0]
        assert "n_batch: int = 20" in source_pix_1
        assert "n_live=150" in source_pix_1
        source_pix_2 = pipeline.split("def source_pix_2(", 1)[1].split("\ndef ", 1)[0]
        assert "n_batch: int = 20" in source_pix_2
        assert "n_live=75" in source_pix_2
        # Both stages free the AdaptSplit regularization the capture frees.
        assert "regularization_init=al.reg.AdaptSplit" in pipeline
        assert "regularization=al.reg.AdaptSplit" in pipeline


def test__replay_runs_the_library_mode_on_captured_lanes():
    executable = _executable(SUBMIT.read_text())
    for flag in (
        "--lanes captured",
        '--batches "$NPZ"',
        "--captured-pass rate",
        "--captured-pass time",
        "--solver-source library",
        "--solver $SOLVER",
        "--arms both",
    ):
        assert flag in executable, f"the submit does not pass {flag}"
    assert f"CONFIG_NAME={CONFIG_NAME}" in executable
    assert "--certified-budget" not in executable, (
        "the rows must run the library's resolved budget, the value production takes"
    )


def test__submit_guards_against_a_stale_checkout_and_mirror():
    executable = _executable(SUBMIT.read_text())
    assert 'grep -q -- "--captured-pass"' in executable
    assert "predates #304" in executable
    assert "11b93476" in executable, "no guard that the mirror carries PyAutoArray#567"
    assert "PYAUTO_HPC_BASE=/mnt/ral/jnightin/PyAuto" in SUBMIT.read_text()


def test__submit_preserves_footer_when_the_numerical_gate_fails():
    executable = _executable(SUBMIT.read_text())
    python_pos = executable.index("python3 -u")
    census_pos = executable.index("AUTOTUNE_ENTRIES count=")
    assert "SAVED_ERR_TRAP=$(trap -p ERR)" in executable[:python_pos]
    assert "trap - ERR" in executable[:python_pos]
    assert 'eval "$SAVED_ERR_TRAP"' in executable[python_pos:census_pos]
    assert "CELL_EXIT=$CELL_RC" in executable[census_pos:]


def test__submit_starts_from_a_fresh_compilation_cache():
    text = SUBMIT.read_text()
    assert 'rm -rf "$JAX_COMPILATION_CACHE_DIR"' in text
    assert 'rm -rf "$TRACE_DIR"' in text
    assert "SLURM_ARRAY_TASK_ID" in text.split("JAX_COMPILATION_CACHE_DIR=", 1)[1].splitlines()[0]


def test__the_gate_is_pre_registered_and_no_tighter_than_1e_9():
    submit = SUBMIT.read_text()
    cell = CELL.read_text()
    assert "PRE-REGISTERED" in submit
    assert "hst_gpu_residue_phase3_psf_2026_09.md" in submit
    assert "GATED at 1e-9 relative:  the jit(vmap) arm vs the jit(vmap) library PDIP" in submit
    assert "GATED at 1e-9 relative:  the scalar jit arm vs the scalar jit library PDIP" in submit
    assert "LANE_RTOL = 1.0e-9" in cell


def test__the_cell_declares_the_phase_c1_flags():
    cell = CELL.read_text()
    for flag in ("--batches", "--batch-sample", "--captured-pass"):
        assert f'"{flag}"' in cell
    assert '"identical", "captured"' in cell
    for key in ("captured_pass", "by_nautilus_phase", "by_call_half", "vs_capture_pdip"):
        assert f'"{key}"' in cell
    assert "_captured_rate_lib{LIBRARY_SOLVER}" in cell


def test__submit_carries_a_wall_basis_block_that_the_gate_accepts():
    from wall.check_submits import check_text, parse_basis_rows

    text = SUBMIT.read_text()
    rows = parse_basis_rows(text)
    assert {row["cell"] for row in rows} == {
        "imaging/nautilus_batch_capture/hst",
        "imaging/fixed_light_trace/hst",
    }
    assert not check_text(text), check_text(text)
