"""The phase-2c point-source campaign manifest and its submits (autolens_profiling#472).

The manifest must be the 10 planned point-source cells, every one with a task (since #481 the
planner no longer plans the ``jax_gpu_na_fp64@hpc_ral_cpu`` source-plane cells, a GPU setup on a
host class with no GPU). The runtime cell runs the solved likelihood (human ruling 2026-10-29) and
the free-centre runtime runs beside it as an ``experiments/plain_runtime/`` row. The submits must
be what the generator renders, run every cell once (4 on `ral`, 6 in one `--gres=gpu:1` job, the ``jax_cpu@hpc_a100`` legs on the node's CPU),
never put a CPU-only job on the `gpu` partition, pass the wall gate inside the fixed 1:00:00
walls (decision D3) and pin the private checkout at 2026.10.25.1 (decision D1). The 2a and 2b
submits stay byte-identical. Modules are loaded by path; no cell script is run.
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


p2c = _load("point_source_p2c", ROOT / "scripts/misc/campaigns/point_source_p2c.py")
p2b = _load("interferometer_p2b", ROOT / "scripts/misc/campaigns/interferometer_p2b.py")
p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from wall import check_submits  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
import catalogue_lifecycle as lc  # noqa: E402

sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

RAL = "hpc/batch_cpu/submit_campaign_p2c_point_source_ral_cpu_fp64"
A100 = "hpc/batch_gpu/submit_campaign_p2c_point_source_a100_fp64"
ALL = [RAL, A100]


def planned_point_source_cells():
    doc = json.loads((ROOT / "dashboard" / "coverage.json").read_text())
    return {c["cell"] for c in doc["cells"] if c["likelihood"].startswith("point_source")}


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


def test_manifest_is_the_planned_cells():
    tasks = p2c.planned_tasks()
    cells = {t.cell for t in tasks}
    assert len(tasks) == len(cells) == 10
    assert cells == planned_point_source_cells()
    # #481: no GPU setup is planned on the GPU-less RAL CPU host class.
    assert not any("jax_gpu" in c and "hpc_ral_cpu" in c for c in planned_point_source_cells())
    assert Counter(t.host_class for t in tasks) == {"hpc_ral_cpu": 4, "hpc_a100": 6}
    # A GPU setup only ever runs on the A100 host class.
    for t in tasks:
        if t.device == "gpu":
            assert t.host_class == "hpc_a100"
    # D2: the source-plane jax_cpu cells on the A100 host class run on its CPU.
    on_a100_cpu = [t for t in tasks if t.host_class == "hpc_a100" and t.device == "cpu"]
    assert {(t.dataset_type, t.measurement) for t in on_a100_cpu} == {
        ("point_source_source", "runtime"),
        ("point_source_source", "breakdown"),
    }


def test_every_task_names_a_real_script_its_flags_parse_and_its_own_cell():
    for task in p2c.planned_tasks() + p2c.plain_tasks():
        assert (ROOT / "scripts" / f"{task.script}.py").is_file(), task.script
        cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
        assert cli.instrument == "simple"
        assert (
            cli.config_name
            == {
                ("hpc_ral_cpu", "cpu"): "hpc_ral_cpu_fp64",
                ("hpc_a100", "gpu"): "hpc_a100_fp64",
                ("hpc_a100", "cpu"): "hpc_a100_cpu_fp64",
            }[(task.host_class, task.device)]
        )
        assert not cli.use_mixed_precision and cli.source_pixels is None
        facts = lc.mirrored(task.output)
        assert facts is not None
        assert (facts["dataset_type"], facts["variant"], facts["instrument"]) == (
            task.dataset_type,
            task.variant,
            "simple",
        )
        assert (facts["setup_id"], facts["host_class"]) == (task.setup_id, task.host_class)
        assert facts["file_measurement"] == task.measurement
        assert facts["model_variant"] == "default"
        if task.kind == "cell":
            assert facts["location"] == "cell"
            # Ruling 2026-10-29: the runtime cell is the solved likelihood.
            expected = {"runtime": "likelihood_runtime_solved", "breakdown": "likelihood_breakdown"}
            assert task.script.endswith(expected[task.measurement])
        else:
            assert (facts["location"], facts["run"]) == ("experiments", "plain_runtime")
            assert task.script.endswith("likelihood_runtime")
        assert task.platform == ("cuda" if task.device == "gpu" else "cpu")


def test_plain_runtime_runs_beside_every_runtime_cell():
    runtime = {(t.dataset_type, t.setup_id, t.host_class) for t in p2c.planned_tasks()}
    plain = {(t.dataset_type, t.setup_id, t.host_class) for t in p2c.plain_tasks()}
    assert plain == runtime and len(p2c.plain_tasks()) == 5


# -------------------------------------------------------------------------------- submits


def test_submits_are_what_the_generator_renders():
    assert p2c.main(["--check"]) == 0
    assert set(p2c.submits()) == set(ALL)


def test_phase_2a_and_2b_submits_stay_byte_identical():
    assert p2a.main(["--check"]) == 0
    assert p2b.main(["--check"]) == 0
    for module in (p2a, p2b):
        for rel, text in module.submits().items():
            assert (ROOT / rel).read_text() == text, rel


def test_every_cell_runs_once_4_on_ral_and_6_in_one_gpu_job():
    ran = Counter()
    for path in ALL:
        ran.update(legs(path)[1])
    cells = {t.cell for t in p2c.planned_tasks()}
    plain = {t.cell for t in p2c.plain_tasks()}
    assert set(ran.values()) == {1}
    assert set(ran) == cells | plain

    text, names = legs(RAL)
    assert sbatch(text, "partition") == "ral" and "--gres" not in text
    assert sbatch(text, "array") == f"0-{len(names) - 1}%16"
    assert len([n for n in names if n in cells]) == 4 and len(names) == 6

    text, names = legs(A100)
    assert sbatch(text, "partition") == "gpu" and "#SBATCH --gres=gpu:1" in text
    assert sbatch(text, "array") is None  # one job, the legs in turn
    assert len([n for n in names if n in cells]) == 6 and len(names) == 9
    platforms = dict(zip(names, bash_array(text, "PLATFORMS")))
    for task in p2c.a100_tasks():
        assert platforms[task.cell] == ("cuda" if task.device == "gpu" else "cpu")


def test_every_run_writes_to_the_scratch_tree_and_marks_each_leg_start():
    for path in ALL:
        text = (ROOT / path).read_text()
        root = re.search(r'^RESULTS_ROOT="(\S+)"', text, re.M).group(1)
        assert root == "$AP_ROOT/output/p2c", path
        assert "output/p2c_logs/walls.tsv" in text and "printf 'START" in text
        for out in bash_array(text, "OUTS"):
            assert out.startswith("results/point_source_")


def test_no_cpu_only_job_on_the_gpu_partition():
    for path in ALL:
        text = (ROOT / path).read_text()
        if sbatch(text, "partition") == "gpu":
            assert "#SBATCH --gres=gpu:1" in text, path
        else:
            assert sbatch(text, "partition") == "ral", path


def test_walls_are_one_hour_and_hold_the_laptop_walls():
    """D3: 1:00:00 on both submits, holding WALL_HEADROOM x (the slowest array leg, or the sum
    of the A100 job's legs) plus the slack; every WALL-BASIS row passes the gate."""
    for path, tasks, sequential in (
        (RAL, p2c.ral_cpu_tasks(), False),
        (A100, p2c.a100_tasks(), True),
    ):
        text = (ROOT / path).read_text()
        assert sbatch(text, "time") == "1:00:00", path
        assert check_submits.check_text(text) == [], path
        assert check_submits.parse_slurm_time("1:00:00") >= p2c.needed_s(
            tasks, sequential=sequential
        ), path
        rows = check_submits.parse_basis_rows(text)
        cells, instruments = check_submits.cells_run(text)
        assert {tuple(r["cell"].split("/")[:2]) for r in rows} == cells, path
        assert instruments == {"simple"}
        for row in rows:
            assert row["source"] == "measured-wall" and row["ref"] == p2c.LAPTOP_REF
            assert float(row["headroom"]) >= 1.25
    assert set(p2c.LAPTOP_WALLS) == {t.script for t in p2c.planned_tasks() + p2c.plain_tasks()}


def test_memory():
    assert sbatch((ROOT / RAL).read_text(), "mem") == "16gb"
    assert sbatch((ROOT / A100).read_text(), "mem") == "32gb"
    assert 10 * p2c.LAPTOP_PEAK_RSS_GB < min(p2c.MEM_GB.values())


def test_every_submit_pins_the_private_checkout():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert "export P2C_PIN=2026.10.25.1" in text and p2c.PIN == p2b.PIN == "2026.10.25.1"
        assert p2c.PRIVATE_BASE == p2b.PRIVATE_BASE and p2c.HPC_BASE == p2b.HPC_BASE
        assert "/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1" in text
        assert "REFUSING" in text and "not tag {pin}" in text and "check_revisions" in text
        assert 'AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"' in text
        assert "export JAX_ENABLE_X64=True" in text
        assert "/mnt/ral/jnightin/PyAuto/PyAuto" not in text
        assert "HPCPullPyAuto" not in text
        assert "from point_source_p2c import prepare_dataset" in text
        assert "output/numba_cache/p2c_" in text and "output/jax_cache/p2c_" in text
        assert "WALL_BASIS_STATUS" not in text  # no batch 0 / batch 1 split (D3)


def test_prepare_dataset_needs_the_committed_dataset(tmp_path):
    p2c.prepare_dataset("simple")  # committed in the repo
    try:
        p2c.prepare_dataset("simple", root=tmp_path)
    except SystemExit as exc:
        assert "missing committed point-source dataset" in str(exc)
    else:
        raise AssertionError("prepare_dataset accepted a missing dataset")
