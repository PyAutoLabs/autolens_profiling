"""The phase-2g weak-lensing campaign manifest and its submits (autolens_profiling#476).

The manifest must be exactly the planned weak cells (the registry's ``weak/isothermal/simple`` on
the §3.3 weak row, whose GPU setup is planned on the GPU host only since #481, decision D5), all
of them runnable; the submits must be what the generator renders, run every runnable cell once
(2 on ``ral``, 4 in one ``--gres=gpu:1`` job), pass the wall gate, pin the private checkout at
2026.10.25.1, and keep batch 1 behind the provisional-wall guard. The 2a and 2b submits stay
byte-identical on the shared machinery. Modules are loaded by path; no cell script is run.
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


p2g = _load("weak_p2g", ROOT / "scripts/misc/campaigns/weak_p2g.py")
p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")
p2b = _load("interferometer_p2b", ROOT / "scripts/misc/campaigns/interferometer_p2b.py")
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from wall import check_submits, rates  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
import catalogue_lifecycle as lc  # noqa: E402


def assert_measured_every_designated_cell(manifest, planned):
    """A finished phase-2 manifest measured every cell the day-zero designations plan
    (autolens_profiling#497); each extra cell is a setup its designation did not name."""
    manifest, planned = set(manifest), set(planned)
    assert planned <= manifest, sorted(planned - manifest)
    for cell in sorted(manifest - planned):
        dataset_type, variant, _instrument = cell.split(" ")[0].split("/")
        setup_id, host_class = cell.split(" ")[1].split("@")
        assert lc.designation(dataset_type, variant, host_class, setup_id) is None, cell


sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

CPU = "hpc/batch_cpu/submit_campaign_p2g_"
GPU = "hpc/batch_gpu/submit_campaign_p2g_"
BATCH0_RAL = f"{CPU}batch0_ral_cpu_fp64"
BATCH0_A100 = f"{GPU}batch0_a100_fp64"
RAL1 = f"{CPU}weak_ral_cpu_simple_fp64"
A1001 = f"{GPU}weak_a100_simple_fp64"
BATCH0 = [BATCH0_RAL, BATCH0_A100]
BATCH1 = [RAL1, A1001]
ALL = BATCH0 + BATCH1


def planned_weak_cells():
    doc = json.loads((ROOT / "dashboard" / "coverage.json").read_text())
    return {c["cell"]: c["class"] for c in doc["cells"] if c["likelihood"].startswith("weak/")}


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


def test_manifest_is_exactly_the_planned_weak_cells():
    planned = planned_weak_cells()
    cells = p2g.planned_cells()
    assert_measured_every_designated_cell({t.cell for t in cells}, planned)
    assert len(cells) == 6
    assert not any(
        t.setup_id == "jax_gpu_dense_fp64" and t.host_class == "hpc_ral_cpu" for t in cells
    )
    # Day zero: the CPU node's optimal is jax_cpu, the A100 node's is jax_gpu; no secondary.
    for t in cells:
        if t.cell not in planned:
            continue
        expected = "optimal"
        assert planned[t.cell] == expected
        assert t.setup_id == (
            "jax_gpu_dense_fp64" if t.host_class == "hpc_a100" else "jax_cpu_dense_fp64"
        )
        assert lc.planned_setups("weak", t.variant, t.host_class, t.instrument)[t.setup_id] == (
            expected
        )


def test_every_planned_cell_is_runnable():
    tasks = p2g.planned_tasks()
    assert len(tasks) == 6
    assert {t.cell for t in tasks} == {t.cell for t in p2g.planned_cells()}
    assert Counter(t.host_class for t in tasks) == {"hpc_ral_cpu": 2, "hpc_a100": 4}
    # D4: the CPU setup runs on the A100 host's CPU.
    for t in tasks:
        assert t.platform == ("cuda,cpu" if t.device == "gpu" else "cpu")


def test_every_task_names_a_real_script_its_flags_parse_and_its_own_cell():
    for task in p2g.planned_tasks():
        assert (ROOT / "scripts" / f"{task.script}.py").is_file(), task.script
        cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
        assert cli.instrument == task.instrument
        assert (
            cli.config_name
            == {"hpc_ral_cpu": "hpc_ral_cpu_fp64", "hpc_a100": "hpc_a100_fp64"}[task.host_class]
        )
        facts = lc.mirrored(task.output)
        assert facts is not None and facts["location"] == "cell"
        assert (facts["dataset_type"], facts["variant"], facts["instrument"]) == (
            "weak",
            task.variant,
            task.instrument,
        )
        assert (facts["setup_id"], facts["host_class"], facts["file_measurement"]) == (
            task.setup_id,
            task.host_class,
            task.measurement,
        )


def test_the_a100_job_runs_the_cpu_setup_first():
    order = [(t.device, t.measurement) for t in p2g.a100_tasks()]
    assert order == [
        ("cpu", "runtime"),
        ("cpu", "breakdown"),
        ("gpu", "runtime"),
        ("gpu", "breakdown"),
    ]


def test_the_runtime_script_takes_the_headline_median():
    assert "scripts/weak/isothermal/likelihood_runtime.py" in rates.HEADLINE_MEDIAN_SCRIPTS


# -------------------------------------------------------------------------------- submits


def test_submits_are_what_the_generator_renders():
    assert p2g.main(["--check"]) == 0
    assert set(p2g.submits()) == set(ALL)


def test_phase_2a_and_2b_submits_stay_byte_identical_on_the_shared_machinery():
    assert p2a.main(["--check"]) == 0
    assert p2b.main(["--check"]) == 0


def test_each_batch_runs_every_runnable_cell_once():
    for batch in (BATCH0, BATCH1):
        ran = Counter()
        for path in batch:
            ran.update(legs(path)[1])
        assert set(ran.values()) == {1}
        assert set(ran) == {t.cell for t in p2g.planned_tasks()}
    for path in (BATCH0_RAL, RAL1):
        text, names = legs(path)
        assert sbatch(text, "partition") == "ral" and "--gres" not in text
        assert sbatch(text, "array") == f"0-{len(names) - 1}%8" and len(names) == 2
    for path in (BATCH0_A100, A1001):
        text, names = legs(path)
        assert sbatch(text, "partition") == "gpu" and "#SBATCH --gres=gpu:1" in text
        assert sbatch(text, "array") is None and len(names) == 4


def test_every_run_writes_to_a_scratch_tree_and_marks_each_leg_start():
    for path in ALL:
        text = (ROOT / path).read_text()
        root = re.search(r'^RESULTS_ROOT="(\S+)"', text, re.M).group(1)
        batch = 0 if path in BATCH0 else 1
        assert root == f"$AP_ROOT/output/p2g_batch{batch}", path
        assert "output/p2g_logs/walls.tsv" in text and "printf 'START" in text
        for out in bash_array(text, "OUTS"):
            assert out.startswith("results/weak/isothermal/simple/")


def test_every_submit_passes_the_wall_gate():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert check_submits.check_text(text) == [], path
        rows = check_submits.parse_basis_rows(text)
        cells, instruments = check_submits.cells_run(text)
        assert {tuple(r["cell"].split("/")[:2]) for r in rows} == cells, path
        assert cells == {("weak", "likelihood_runtime"), ("weak", "likelihood_breakdown")}
        assert instruments == {"simple"}


def test_batch1_walls_are_measured_by_batch0_and_batch0_never_guards():
    """Every rerun row cites batch 0 (RAL 413293, A100 job 413297): one instrument, so every leg
    bounds itself; --time holds 1.25x the slowest RAL leg or the A100 job's sum, plus slack."""
    assert p2g.WALL_BASIS_STATUS == "measured"
    assert set(p2g.BATCH0_WALLS) == {t.cell for t in p2g.planned_tasks(0)}
    assert set(p2g.BATCH0_LIMITATIONS) <= set(p2g.BATCH0_WALLS)
    for path in BATCH1:
        text, names = legs(path)
        assert "WALL_BASIS_STATUS=measured" in text
        assert text.index("WALL_BASIS_STATUS=measured") < text.index(
            'source "$AP_ROOT/activate.sh"'
        )
        tasks = [t for t in p2g.planned_tasks() if t.cell in set(names)]
        assert len(tasks) == len(names), path
        walls = []
        for t in tasks:
            basis = p2g.basis_leg(t)
            assert basis.cell in p2g.BATCH0_WALLS
            assert basis.script == t.script and basis.setup_id == t.setup_id
            wall, elapsed, _ = p2g.BATCH0_WALLS[basis.cell]
            assert p2g.measured_wall(t)[0] == max(wall, elapsed)
            walls.append(max(wall, elapsed))
        need = 1.25 * (sum(walls) if path in (A1001,) else max(walls)) + p2g.WALL_MIN_SLACK_S
        assert check_submits.parse_slurm_time(sbatch(text, "time")) >= need, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "measured-wall", path
    # Peak MaxRSS 0.65 GiB: the declared --mem stands.
    mems = {p: sbatch((ROOT / p).read_text(), "mem") for p in ALL}
    assert mems == {BATCH0_RAL: "16gb", RAL1: "16gb", BATCH0_A100: "32gb", A1001: "32gb"}
    for path in BATCH0:
        text = (ROOT / path).read_text()
        assert "WALL_BASIS_STATUS" not in text and "ALLOW_PROVISIONAL" not in text, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "unmeasured" and float(row["headroom"]) >= 3.0


def test_every_submit_pins_the_private_checkout():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert "export P2G_PIN=2026.10.25.1" in text and p2g.PIN == "2026.10.25.1"
        assert "/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1" in text
        assert "REFUSING" in text and "not tag {pin}" in text and "check_revisions" in text
        assert 'AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"' in text
        assert "/mnt/ral/jnightin/PyAuto/PyAuto" not in text and "HPCPullPyAuto" not in text
        assert "from weak_p2g import prepare_dataset" in text
        assert "output/numba_cache/p2g_" in text and "output/jax_cache/p2g_" in text
        assert "Decisions adopted by the human" in text


def test_prepare_dataset_returns_early_when_the_catalogue_exists(tmp_path):
    folder = tmp_path / "dataset" / "weak" / "simple"
    folder.mkdir(parents=True)
    (folder / "dataset.json").write_text("{}")
    p2g.prepare_dataset("simple", root=tmp_path)  # no simulation, no import of the stack
    assert (tmp_path / "output" / "p2g_dataset_simple.lock").exists()
