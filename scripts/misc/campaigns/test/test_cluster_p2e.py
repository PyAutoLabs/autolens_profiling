"""The phase-2e cluster campaign manifest and its submits (autolens_profiling#474).

The manifest must be exactly the planned cluster cells of ``dashboard/coverage.json`` (image_plane
from the registry, source_plane drafted by analogy with point_source_source, decision D2), with
no cell a RAL job cannot run (the planner plans GPU setups only on GPU hosts, #481); the submits must be what the generator renders, run
every runnable cell once (4 on `ral`, 6 in a `--gres=gpu:1` job), never put a CPU-only job on the
`gpu` partition, pass the wall gate, pin the private checkout at 2026.10.25.1 and refuse batch 1
while the walls are provisional. The shared machinery must leave the 2a and 2b submits
byte-identical. Modules are loaded by path; no cell script is run.
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


p2e = _load("cluster_p2e", ROOT / "scripts/misc/campaigns/cluster_p2e.py")
p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")
p2b = _load("interferometer_p2b", ROOT / "scripts/misc/campaigns/interferometer_p2b.py")
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from wall import check_submits  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
import catalogue_lifecycle as lc  # noqa: E402

sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

CPU = "hpc/batch_cpu/submit_campaign_p2e_"
GPU = "hpc/batch_gpu/submit_campaign_p2e_"
RAL1 = f"{CPU}cluster_ral_cpu_fp64"
A1001 = f"{GPU}cluster_a100_fp64"
BATCH0_RAL = f"{CPU}batch0_ral_cpu_fp64"
BATCH0_A100 = f"{GPU}batch0_a100_fp64"
BATCH1 = [RAL1, A1001]
BATCH0 = [BATCH0_RAL, BATCH0_A100]
ALL = BATCH1 + BATCH0


def planned_cluster_cells():
    doc = json.loads((ROOT / "dashboard" / "coverage.json").read_text())
    return {c["cell"] for c in doc["cells"] if c["likelihood"].startswith("cluster/")}


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


def test_manifest_is_exactly_the_planned_cluster_cells():
    cells = p2e.coverage_cells()
    assert len(cells) == len({t.cell for t in cells}) == 10
    assert {t.cell for t in cells} == planned_cluster_cells()
    assert Counter(t.variant for t in cells) == {"image_plane": 4, "source_plane": 6}
    # The planned setups are the §3.3 tables the catalogue routes each variant to (D2).
    for variant, table in (
        ("image_plane", "point_source_image"),
        ("source_plane", "point_source_source"),
    ):
        for host in ("hpc_ral_cpu", "hpc_a100"):
            planned = set(lc.planned_setups("cluster", variant, host, "simple"))
            assert planned == {s for s, h in p2e.SETUPS[variant] if h == host}, (variant, host)
        assert lc._table_name("cluster", variant) == table


def test_no_gpu_setup_is_planned_on_the_cpu_host_so_every_cell_runs():
    # The planner plans a GPU setup only on a gpu_hosts host (#481), which settled D4.
    assert p2e.unrunnable_tasks() == []
    assert not any(
        t.device == "gpu" and t.host_class == "hpc_ral_cpu" for t in p2e.coverage_cells()
    )
    tasks = p2e.planned_tasks()
    assert len(tasks) == 10 and tasks == p2e.coverage_cells()
    assert Counter(t.host_class for t in tasks) == {"hpc_ral_cpu": 4, "hpc_a100": 6}
    for t in tasks:
        if t.host_class == "hpc_ral_cpu":
            assert t.device == "cpu"


def test_every_task_names_a_real_script_its_flags_parse_and_its_own_cell():
    for task in p2e.planned_tasks():
        assert (ROOT / "scripts" / f"{task.script}.py").is_file(), task.script
        cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
        assert cli.instrument == "simple" and cli.standard_size
        assert cli.source_pixels is None and not cli.use_sparse_operator
        assert (
            cli.config_name
            == {"hpc_ral_cpu": "hpc_ral_cpu_fp64", "hpc_a100": "hpc_a100_fp64"}[task.host_class]
        )
        facts = lc.mirrored(task.output)
        assert facts is not None and facts["location"] == "cell"
        assert (facts["dataset_type"], facts["variant"], facts["instrument"]) == (
            "cluster",
            task.variant,
            "simple",
        )
        assert (facts["setup_id"], facts["host_class"], facts["file_measurement"]) == (
            task.setup_id,
            task.host_class,
            task.measurement,
        )
        assert facts["model_variant"] == "default"
        # A jax_cpu leg runs on the CPU even inside the A100 job.
        assert task.platform == ("cuda,cpu" if task.device == "gpu" else "cpu")
        assert task.gate_cell == f"cluster/{task.variant}"


def test_the_standard_solver_is_the_decided_workspace_solver():
    sys.path.insert(0, str(ROOT / "scripts" / "misc"))
    from likelihood_breakdown import cluster_standard as cs

    assert cs.STANDARD_SOLVER == {
        "shape_native": (100, 100),
        "pixel_scales": 1.0,
        "pixel_scale_precision": 0.001,
        "magnification_threshold": 0.1,
        "residual_threshold": 10.0,
    }
    assert cs.solver_settings(True) == cs.STANDARD_SOLVER
    assert cs.solver_settings(False)["shape_native"] == (200, 200)
    record = cs.settings_record(cs.STANDARD_SOLVER)
    assert record["residual_threshold_source"] == "explicit"
    assert cs.settings_record(cs.HISTORIC_SOLVER)["residual_threshold_source"] == "library default"
    assert set(cs.FIT_CLASS_NAMES) == set(cs.LIKELIHOODS) == set(p2e.VARIANTS)


# -------------------------------------------------------------------------------- submits


def test_submits_are_what_the_generator_renders():
    assert p2e.main(["--check"]) == 0
    assert set(p2e.submits()) == set(ALL)


def test_phase_2a_and_2b_submits_stay_byte_identical_on_the_shared_machinery():
    for module in (p2a, p2b):
        assert module.main(["--check"]) == 0
        for rel, text in module.submits().items():
            assert (ROOT / rel).read_text() == text, rel


def test_batch1_runs_every_runnable_cell_once_4_on_ral_and_6_on_gpu():
    ran = Counter()
    for path in BATCH1:
        text, names = legs(path)
        ran.update(names)
        if path == RAL1:
            assert sbatch(text, "partition") == "ral"
            assert sbatch(text, "array") == f"0-{len(names) - 1}%8"
        else:
            assert sbatch(text, "partition") == "gpu" and "#SBATCH --gres=gpu:1" in text
            assert sbatch(text, "array") is None
    assert set(ran.values()) == {1}
    assert set(ran) == {t.cell for t in p2e.planned_tasks()}
    assert (len(legs(RAL1)[1]), len(legs(A1001)[1])) == (4, 6)
    platforms = dict(
        zip(*(bash_array((ROOT / A1001).read_text(), k) for k in ("CELLNAMES", "PLATFORMS")))
    )
    assert Counter(platforms.values()) == {"cuda,cpu": 4, "cpu": 2}


def test_batch0_is_the_image_plane_slice():
    _, ral = legs(BATCH0_RAL)
    _, a100 = legs(BATCH0_A100)
    assert (len(ral), len(a100)) == (2, 2)
    for task in p2e.batch0_ral_cpu() + p2e.batch0_a100():
        assert task.variant == "image_plane" and task.batch == 0
    text = (ROOT / BATCH0_A100).read_text()
    assert sbatch(text, "array") == "0-1%2" and "#SBATCH --gres=gpu:1" in text
    assert sbatch((ROOT / BATCH0_RAL).read_text(), "array") == "0-1%8"
    # Every batch-1 leg is bounded by a batch-0 leg of the same measurement and device.
    batch0 = {t.cell for t in p2e.batch0_ral_cpu() + p2e.batch0_a100()}
    for task in p2e.planned_tasks():
        basis = p2e.basis_leg(task)
        assert basis.cell in batch0, task.cell
        assert (basis.measurement, basis.device) == (task.measurement, task.device)


def test_every_run_writes_to_a_scratch_tree_and_marks_each_leg_start():
    for path in ALL:
        text = (ROOT / path).read_text()
        root = re.search(r'^RESULTS_ROOT="(\S+)"', text, re.M).group(1)
        assert root.startswith("$AP_ROOT/output/p2e_batch"), path
        assert "output/p2e_logs/walls.tsv" in text and "printf 'START" in text
        for out in bash_array(text, "OUTS"):
            assert out.startswith("results/cluster/")


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


def test_walls_are_provisional_and_batch1_refuses_until_batch0_lands():
    """DRAFT: nothing has run. Batch 1 renders `unmeasured` rows and refuses to run while
    WALL_BASIS_STATUS is provisional; batch 0 never guards."""
    assert p2e.WALL_BASIS_STATUS == "provisional"
    assert p2e.BATCH0_WALLS == {} and p2e.BATCH0_PEAK_RSS_KIB == {}
    for path in BATCH1:
        text = (ROOT / path).read_text()
        assert "WALL_BASIS_STATUS=provisional" in text and "P2E_ALLOW_PROVISIONAL_WALL" in text
        assert text.index("WALL_BASIS_STATUS=provisional") < text.index(
            'source "$AP_ROOT/activate.sh"'
        )
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "unmeasured" and row["probe-first"] == "yes"
    times = {p: sbatch((ROOT / p).read_text(), "time") for p in ALL}
    assert times == {
        RAL1: p2e.BATCH1_PROVISIONAL_TIME["ral_cpu"],
        A1001: p2e.BATCH1_PROVISIONAL_TIME["a100"],
        BATCH0_RAL: p2e.BATCH0_TIME["ral_cpu"],
        BATCH0_A100: p2e.BATCH0_TIME["a100"],
    }
    for path in BATCH0:
        text = (ROOT / path).read_text()
        assert "WALL_BASIS_STATUS" not in text and "ALLOW_PROVISIONAL" not in text, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "unmeasured" and float(row["headroom"]) >= 3.0
    for path in ALL:
        assert sbatch((ROOT / path).read_text(), "mem") == "32gb", path


def test_every_submit_pins_the_private_checkout():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert "export P2E_PIN=2026.10.25.1" in text and p2e.PIN == "2026.10.25.1"
        assert "/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1" in text
        assert "REFUSING" in text and "not tag {pin}" in text and "check_revisions" in text
        assert 'AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"' in text
        assert "/mnt/ral/jnightin/PyAuto/PyAuto" not in text
        assert "HPCPullPyAuto" not in text
        assert "from cluster_p2e import prepare_dataset" in text
        assert "output/numba_cache/p2e_" in text and "output/jax_cache/p2e_" in text
