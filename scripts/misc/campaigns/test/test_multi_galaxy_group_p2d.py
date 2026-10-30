"""The phase-2d multi_galaxy + group campaign manifest and its submits (autolens_profiling#473).

The manifest must be exactly the 48 planned multi_galaxy and group cells (the draft decisions
D1, D2, D6); the submits must be what the generator renders, run every cell once in batch 1 (20
on `ral`, 28 in `--gres=gpu:1` jobs), run every group leg in batch 0 (D9), never put a CPU-only
job on the `gpu` partition, pass the wall gate, pin the private checkout at 2026.10.25.1, and
refuse batch 1 while its walls are provisional. The shared machinery must leave the 2a and 2b
submits byte-identical. Modules are loaded by path; no cell script is run.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p2d = _load("multi_galaxy_group_p2d", ROOT / "scripts/misc/campaigns/multi_galaxy_group_p2d.py")
p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")
p2b = _load("interferometer_p2b", ROOT / "scripts/misc/campaigns/interferometer_p2b.py")
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from wall import check_submits, rates  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
import catalogue_lifecycle as lc  # noqa: E402

sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

CPU = "hpc/batch_cpu/submit_campaign_p2d_"
GPU = "hpc/batch_gpu/submit_campaign_p2d_"
RAL1 = [f"{CPU}multi_galaxy_ral_cpu_hst_fp64", f"{CPU}group_ral_cpu_euclid_fp64"]
A1001 = [f"{GPU}multi_galaxy_a100_hst_fp64", f"{GPU}group_a100_euclid_fp64"]
BATCH1 = RAL1 + A1001
BATCH0_RAL = f"{CPU}batch0_ral_cpu_fp64"
BATCH0_A100 = f"{GPU}batch0_a100_fp64"
BATCH0 = [BATCH0_RAL, BATCH0_A100]
ALL = BATCH1 + BATCH0


def planned_cells():
    doc = json.loads((ROOT / "dashboard" / "coverage.json").read_text())
    return {
        c["cell"]
        for c in doc["cells"]
        if c["likelihood"].split("/")[0] in ("multi_galaxy", "group")
    }


def bash_array(text, name):
    match = re.search(rf"^{name}=\(\n(.*?)\n\)", text, re.M | re.S)
    assert match, name
    return [shlex.split(line)[0] if line.strip() else "" for line in match.group(1).splitlines()]


def sbatch(text, key):
    match = re.search(rf"^#SBATCH --{key}=(\S+)", text, re.M)
    return match.group(1) if match else None


def legs(path):
    text = (ROOT / path).read_text()
    names = bash_array(text, "CELLNAMES")
    for k in ("CELLS", "INSTS", "FLAGS", "PLATFORMS", "OUTS"):
        assert len(bash_array(text, k)) == len(names), (path, k)
    return text, names


# ------------------------------------------------------------------------------- manifest


def test_manifest_is_exactly_the_planned_cells():
    tasks = p2d.planned_tasks()
    assert len(tasks) == len({t.cell for t in tasks}) == 48
    assert {t.cell for t in tasks} == planned_cells()
    assert Counter(t.host_class for t in tasks) == {"hpc_ral_cpu": 20, "hpc_a100": 28}
    assert Counter((t.dataset_type, t.variant) for t in tasks) == {
        (dt, v): n
        for dt in ("multi_galaxy", "group")
        for v, n in (("mge", 4), ("rectangular", 10), ("delaunay", 10))
    }
    assert {(t.dataset_type, t.instrument) for t in tasks} == {
        ("multi_galaxy", "hst"),
        ("group", "euclid"),
    }


def test_the_setups_are_the_33_imaging_tables():
    for t in p2d.planned_tasks():
        planned = lc.planned_setups(t.dataset_type, t.variant, t.host_class, t.instrument)
        assert t.setup_id in planned, t.cell
        table = "pixelized" if t.variant in p2d.PIXELIZED else "parametric"
        assert t.setup_id in p2d.SETUPS[table][t.host_class]
    registry = json.loads((ROOT / "catalogue" / "registry.json").read_text())
    declared = {
        (c["dataset"], c["model"], c["instrument"])
        for c in registry["cells"]
        if c["dataset"] in ("multi_galaxy", "group")
    }
    assert declared == {(t.dataset_type, t.variant, t.instrument) for t in p2d.planned_tasks()}


def test_every_task_names_a_real_script_its_flags_parse_and_its_own_cell():
    for task in p2d.planned_tasks():
        script = ROOT / "scripts" / f"{task.script}.py"
        assert script.is_file(), task.script
        text = script.read_text()
        engine = "numba" if task.backend == "numba" else "jax"
        assert (
            f'_driver.main("{task.dataset_type}", "{task.variant}", "{task.measurement}", '
            f'"{engine}", script_file=__file__)'
        ) in text
        if engine == "numba":  # threads pinned before the driver (and numpy) is imported
            assert text.index("\n_pin_thread_env(1)") < text.index("import galaxy_group_standard")
        cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
        assert cli.instrument == task.instrument
        assert cli.use_sparse_operator == (task.backend == "jax" and task.operator == "sparse")
        assert cli.config_name == p2d.CONFIG_NAMES[(task.host_class, task.device)]
        facts = lc.mirrored(task.output)
        assert facts is not None and facts["location"] == "cell"
        assert (facts["dataset_type"], facts["variant"], facts["instrument"]) == (
            task.dataset_type,
            task.variant,
            task.instrument,
        )
        assert (facts["setup_id"], facts["host_class"], facts["file_measurement"]) == (
            task.setup_id,
            task.host_class,
            task.measurement,
        )
        assert task.platform == ("cuda,cpu" if task.device == "gpu" else "cpu")


def test_the_jax_runtime_cells_take_the_headline_median_in_the_wall_gate():
    runtimes = {
        f"scripts/{t.script}.py"
        for t in p2d.planned_tasks()
        if t.backend == "jax" and t.measurement == "runtime"
    }
    assert len(runtimes) == 6 and runtimes <= rates.HEADLINE_MEDIAN_SCRIPTS


# -------------------------------------------------------------------------------- submits


def test_submits_are_what_the_generator_renders():
    assert p2d.main(["--check"]) == 0
    assert set(p2d.submits()) == set(ALL)


def test_phase_2a_and_2b_submits_stay_byte_identical_on_the_shared_machinery():
    assert p2a.main(["--check"]) == 0
    assert p2b.main(["--check"]) == 0


def test_batch1_runs_every_cell_once_20_on_ral_and_28_on_gpu():
    ran = Counter()
    for path in BATCH1:
        text, names = legs(path)
        ran.update(names)
        assert len(names) == (10 if path in RAL1 else 14)
        if path in RAL1:
            assert sbatch(text, "partition") == "ral"
            assert sbatch(text, "array") == f"0-{len(names) - 1}%16"
        else:
            assert sbatch(text, "partition") == "gpu" and "#SBATCH --gres=gpu:1" in text
            assert sbatch(text, "array") is None
    assert set(ran.values()) == {1}
    assert set(ran) == {t.cell for t in p2d.planned_tasks()}


def test_batch0_is_every_group_leg():
    _, ral = legs(BATCH0_RAL)
    _, a100 = legs(BATCH0_A100)
    assert (len(ral), len(a100)) == (10, 14)
    assert set(ral) | set(a100) == {
        t.cell for t in p2d.planned_tasks() if t.dataset_type == "group"
    }
    text = (ROOT / BATCH0_A100).read_text()
    assert sbatch(text, "array") == "0-13%2" and "#SBATCH --gres=gpu:1" in text
    assert sbatch((ROOT / BATCH0_RAL).read_text(), "array") == "0-9%8"
    for path in BATCH0:
        text = (ROOT / path).read_text()
        assert sbatch(text, "time") == "4:00:00" and sbatch(text, "mem") == "32gb"
        assert "WALL_BASIS_STATUS" not in text and "ALLOW_PROVISIONAL" not in text
    # Every batch-1 leg has a batch-0 leg of the same script, setup and host class.
    batch0 = {t.cell for t in p2d.batch0_ral_cpu() + p2d.batch0_a100()}
    for t in p2d.planned_tasks():
        basis = p2d.basis_leg(t)
        assert basis.cell in batch0 and basis.script.split("/", 1)[1] == t.script.split("/", 1)[1]


def test_batch1_walls_are_measured_by_batch0_and_batch0_never_guards():
    """Every batch-1 row cites batch 0 (RAL 413272, A100 413294) on the bounding group leg of the
    same script, setup and host class; --time holds 1.25x the slowest leg of a RAL array or the
    sum of an A100 job's legs, plus the slack."""
    assert p2d.WALL_BASIS_STATUS == "measured"
    assert set(p2d.BATCH0_WALLS) == {t.cell for t in p2d.batch0_ral_cpu() + p2d.batch0_a100()}
    assert set(p2d.BATCH0_LIMITATIONS) <= set(p2d.BATCH0_WALLS)
    for path in BATCH1:
        text, names = legs(path)
        assert "WALL_BASIS_STATUS=measured" in text
        assert text.index("WALL_BASIS_STATUS=measured") < text.index(
            'source "$AP_ROOT/activate.sh"'
        )
        tasks = [t for t in p2d.planned_tasks() if t.cell in set(names)]
        assert len(tasks) == len(names), path
        walls = []
        for t in tasks:
            basis = p2d.basis_leg(t)
            assert basis.cell in p2d.BATCH0_WALLS
            # D9: the group leg of the same variant, setup, host class and measurement.
            assert basis.dataset_type == p2d.BATCH0_DATASET_TYPE
            assert (basis.variant, basis.setup_id, basis.host_class, basis.measurement) == (
                t.variant,
                t.setup_id,
                t.host_class,
                t.measurement,
            )
            wall, elapsed, _ = p2d.BATCH0_WALLS[basis.cell]
            assert p2d.measured_wall(t)[0] == max(wall, elapsed)
            walls.append(max(wall, elapsed))
        need = 1.25 * (sum(walls) if path in A1001 else max(walls)) + p2d.WALL_MIN_SLACK_S
        assert check_submits.parse_slurm_time(sbatch(text, "time")) >= need, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "measured-wall", path
    # --mem: the next step >= 1.5x batch 0's peak MaxRSS on the device, at least 16 GB.
    for path in BATCH1:
        device = "ral_cpu" if path in RAL1 else "a100"
        peak_gib = p2d.BATCH0_PEAK_RSS_KIB[device][0] / 2**20
        mem = int(sbatch((ROOT / path).read_text(), "mem").removesuffix("gb"))
        assert mem >= max(16, 1.5 * peak_gib) and mem == p2d.mem_gb_measured(device), path
    for path in BATCH0:
        text = (ROOT / path).read_text()
        assert "WALL_BASIS_STATUS" not in text and "ALLOW_PROVISIONAL" not in text, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "unmeasured" and float(row["headroom"]) >= 3.0


def test_every_run_writes_to_a_scratch_tree_and_marks_each_leg_start():
    for path in ALL:
        text = (ROOT / path).read_text()
        root = re.search(r'^RESULTS_ROOT="(\S+)"', text, re.M).group(1)
        assert root in ("$AP_ROOT/output/p2d_batch0", "$AP_ROOT/output/p2d_batch1"), path
        assert "output/p2d_logs/walls.tsv" in text and "printf 'START" in text
        for out in bash_array(text, "OUTS"):
            assert out.split("/")[1] in ("multi_galaxy", "group")
        assert "Decisions D1-D12 adopted" in text


def test_no_cpu_only_job_on_the_gpu_partition():
    for path in ALL:
        text = (ROOT / path).read_text()
        if sbatch(text, "partition") == "gpu":
            assert "#SBATCH --gres=gpu:1" in text, path
        else:
            assert sbatch(text, "partition") == "ral", path


def test_every_submit_passes_the_wall_gate():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert check_submits.check_text(text) == [], path
        rows = check_submits.parse_basis_rows(text)
        cells, _ = check_submits.cells_run(text)
        assert {tuple(r["cell"].split("/")[:2]) for r in rows} == cells, path


def test_every_submit_pins_the_private_checkout():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert "export P2D_PIN=2026.10.25.1" in text and p2d.PIN == "2026.10.25.1"
        assert "/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1" in text
        assert "REFUSING" in text and "not tag {pin}" in text and "check_revisions" in text
        assert 'AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"' in text
        assert "/mnt/ral/jnightin/PyAuto/PyAuto" not in text and "HPCPullPyAuto" not in text
        assert "from multi_galaxy_group_p2d import prepare_dataset" in text
        assert "output/numba_cache/p2d_" in text and "output/jax_cache/p2d_" in text


def test_prepare_dataset_maps_each_instrument_to_one_dataset_type():
    assert {i: p2d.dataset_type_of(i) for i in ("hst", "euclid")} == {
        "hst": "multi_galaxy",
        "euclid": "group",
    }
