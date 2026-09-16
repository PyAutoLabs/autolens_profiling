"""Static checks for the phase-2 batched ``fixed_light_vmap`` A100 submits (#273).

**No imaging, no JAX, no GPU** — these read the submit script and the cell as text.

These legs run the *same* cell as the phase-1 trace family
(``scripts/imaging/likelihood_breakdown/fixed_light_trace.py``) but in its
``--vmap-batch`` mode, so they are a fourth ``fixed_light`` submit family with
their own file-name prefix (``..._fixed_light_vmap_*``). They are excluded from
``test_fixed_light_cell.py``'s glob for the reason that file states: the config
names nest as substrings, so an un-excluded family passes the wrong file's
assertions **vacuously**. This file is the vmap family's half of that bargain.

What a wrong submit would look like, and what catches it:

- the three arm tables and the ``--array`` range disagree, so a task runs with
  an empty batch size — ``test__array_range_matches_the_arms``;
- an arm passes a batch size, lane family or fallback the cell's argparse would
  reject, which ``parse_known_args`` does **not** reject: it ignores unknown
  flags, so the leg runs and quietly writes a table from different settings —
  ``test__every_arm_is_a_setting_the_cell_accepts``;
- the submit runs against a checkout from **before** #273, which would swallow
  every phase-2 flag, measure a single call and write it to the *phase-1*
  filename, clobbering phase 1's own A100 result with a table that looks right
  — ``test__submit_guards_against_a_checkout_without_the_flags``;
- no identical-lanes control, so nothing ties phase 2 to the 21.4 ms/lane row
  the campaign already owns — ``test__the_array_carries_the_identical_lane_control``.
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

#: The lane families and fallback modes the cell's argparse accepts.
LANE_MODES = ("distinct", "identical")
FALLBACK_MODES = ("on", "off")

#: These legs share phase 1's config name; the ``_vmap<B>_<lanes>_fb<on|off>``
#: suffix the cell adds to the filename is what keeps them disjoint.
CONFIG_NAME = "hpc_a100_fp64_fixed_light_trace"


def _submits() -> list[_Path]:
    return sorted(BATCH_GPU.glob("submit_breakdown_imaging_fixed_light_vmap_*"))


def _arm_table(text: str, name: str) -> list[str]:
    """The values of a ``NAME=(a b c)`` bash array."""
    match = re.search(rf"^{name}=\((.*?)\)", text, re.M)
    return match.group(1).split() if match else []


def _executable(text: str) -> str:
    """The submit with whole-line comments blanked, as check_submits reads it."""
    return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in text.splitlines())


@pytest.fixture(scope="module")
def submits() -> list[_Path]:
    found = _submits()
    assert found, "no submit_breakdown_imaging_fixed_light_vmap_* submits found"
    return found


def test__the_vmap_family_is_excluded_from_the_phase_0_glob():
    """Or its submits pass test_fixed_light_cell.py's assertions by substring luck."""
    other = (ROOT / "scripts" / "misc" / "test" / "test_fixed_light_cell.py").read_text()
    assert '"_fixed_light_vmap_" not in p.name' in other, (
        "test_fixed_light_cell.py no longer excludes the vmap family; its "
        "--config-name assertion would pass on a vmap leg only because "
        "hpc_a100_fp64_fixed_light_trace contains hpc_a100_fp64_fixed_light"
    )


def test__the_cell_declares_every_phase_2_flag():
    """``parse_known_args`` would ignore any of these, silently."""
    text = CELL.read_text()
    for flag in ("--vmap-batch", "--lanes", "--arms", "--draw-seed", "--fallback"):
        assert flag in text, f"cell does not declare {flag}"


def test__the_cell_records_every_phase_2_flag_in_the_results_json():
    """A flag not in ``configuration`` cannot be told from one that was swallowed."""
    text = CELL.read_text()
    for key in ("vmap_batch", "lanes", "arms", "fallback", "draw_seed"):
        assert f'"{key}"' in text, (
            f"the cell never writes configuration.{key}; a leg run by a checkout that "
            f"swallowed the flag would be indistinguishable from one that honoured it"
        )


def test__the_cell_suffixes_the_filename_with_the_batch_settings():
    """Or a B=4 leg overwrites the B=16 leg, or phase 1's file."""
    text = CELL.read_text()
    assert "_vmap{int(VMAP_BATCH)}_{LANES_MODE}_fb" in text, (
        "the output filename no longer carries the batch size, lane family and "
        "fallback mode — legs of this array would clobber each other"
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
def test__every_arm_is_a_setting_the_cell_accepts(path):
    """``parse_known_args`` IGNORES an unknown flag value, so argparse will not catch this."""
    text = path.read_text()
    batches = _arm_table(text, "BATCHES")
    lanesets = _arm_table(text, "LANESETS")
    fallbacks = _arm_table(text, "FALLBACKS")
    assert batches, f"{path.name}: no BATCHES=(...) arm table"
    assert lanesets, f"{path.name}: no LANESETS=(...) arm table"
    assert fallbacks, f"{path.name}: no FALLBACKS=(...) arm table"
    for batch in batches:
        assert batch.isdigit() and int(batch) >= 1, f"{path.name}: batch {batch!r} is not >= 1"
    for lanes in lanesets:
        assert lanes in LANE_MODES, f"{path.name}: lanes {lanes!r} is not one of {LANE_MODES}"
    for fallback in fallbacks:
        assert fallback in FALLBACK_MODES, (
            f"{path.name}: fallback {fallback!r} is not one of {FALLBACK_MODES}"
        )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__array_range_matches_the_arms(path):
    """A task with no arm would run with an empty batch size — after queueing."""
    text = path.read_text()
    tables = {name: _arm_table(text, name) for name in ("BATCHES", "LANESETS", "FALLBACKS")}
    lengths = {name: len(values) for name, values in tables.items()}
    assert len(set(lengths.values())) == 1, (
        f"{path.name}: the arm tables have different lengths {lengths} — they are "
        f"indexed by the same SLURM_ARRAY_TASK_ID"
    )
    n_arms = next(iter(lengths.values()))
    array = re.search(r"^#SBATCH --array=(\d+)-(\d+)", text, re.M)
    assert array, f"{path.name}: no `#SBATCH --array=` line"
    first, last = int(array.group(1)), int(array.group(2))
    assert first == 0, f"{path.name}: array starts at {first}, but the arm tables are 0-indexed"
    assert last == n_arms - 1, f"{path.name}: --array=0-{last} but only {n_arms} arms are defined"
    assert "FATAL: no (batch, lanes, fallback) arm" in text, (
        f"{path.name}: no guard for a task index with no arm"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__the_array_carries_the_identical_lane_control(path):
    """The control is what ties phase 2 to the 21.4 ms/lane row already on record."""
    lanesets = _arm_table(path.read_text(), "LANESETS")
    assert "identical" in lanesets, (
        f"{path.name}: no identical-lanes arm. Without the control there is nothing "
        f"tying this array to the existing whole-likelihood vmap evidence, which was "
        f"measured on identical lanes"
    )
    assert "distinct" in lanesets, (
        f"{path.name}: no distinct-lanes arm — identical lanes share one straggler and "
        f"hide the cost this array exists to measure"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__both_fallback_semantics_are_rows(path):
    """``lax.cond`` becomes ``select`` under vmap and runs BOTH branches."""
    fallbacks = _arm_table(path.read_text(), "FALLBACKS")
    assert set(fallbacks) == set(FALLBACK_MODES), (
        f"{path.name}: fallback arms are {sorted(set(fallbacks))}. Under vmap the PDIP "
        f"fallback's lax.cond becomes a select and both branches run, so fallback-on and "
        f"fallback-off are two different programs and both are rows, never a footnote"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__both_arms_are_run_so_every_leg_is_a_matched_pair(path):
    executable = _executable(path.read_text())
    assert "--arms both" in executable, (
        f"{path.name}: does not pass `--arms both`. A vmap millisecond without the "
        f"scalar millisecond of the same lanes in the same process is not a comparison"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_guards_against_a_checkout_without_the_flags(path):
    """A pre-#273 checkout would swallow every flag and clobber the phase-1 file."""
    executable = _executable(path.read_text())
    assert 'grep -q -- "--vmap-batch"' in executable, (
        f"{path.name}: no guard that the checked-out cell defines --vmap-batch. "
        f"parse_known_args would ignore every phase-2 flag, the cell would measure a "
        f"single call, and it would write it to the PHASE 1 filename"
    )
    assert "This checkout predates #273" in executable, (
        f"{path.name}: the guard does not say what went wrong"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_writes_under_the_trace_config_name(path):
    text = path.read_text()
    assert f"--config-name {CONFIG_NAME}" in text, (
        f"{path.name}: legs must write under --config-name {CONFIG_NAME}"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_does_not_pass_flags_the_cell_would_silently_swallow(path):
    executable = _executable(path.read_text())
    for flag in ("--pins", "--pass-budget", "--dataset", "--vmap_batch"):
        assert flag not in executable, (
            f"{path.name}: passes {flag}, which this cell does not define — "
            f"parse_known_args would ignore it silently"
        )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_keeps_the_serial_host_cpus(path):
    """qhull is serial host work and this array exists to measure it."""
    text = path.read_text()
    assert "--cpus-per-task=4" in text, (
        f"{path.name}: --cpus-per-task is not 4. The qhull callback is serial host work "
        f"and dropping the CPUs would change the very term being measured"
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
def test__submit_guards_the_shared_dataset(path):
    """Five arms auto-simulating the same dataset dir would interleave FITS writes."""
    text = path.read_text()
    assert "dataset/imaging/hst/data.fits" in text, (
        f"{path.name}: no dataset-exists guard; concurrent arms would race the simulator"
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
