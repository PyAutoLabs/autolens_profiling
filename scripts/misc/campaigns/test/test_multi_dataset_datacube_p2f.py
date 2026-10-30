"""The phase-2f multi_dataset + datacube campaign manifest and its submits (autolens_profiling#475).

The manifest must be exactly the planned multi_dataset/delaunay and datacube/delaunay cells (32
under decision D1: 18 on `ral`, 14 on the A100); the submits must be what the generator renders,
run every cell once, never put a CPU-only job on the `gpu` partition, pass the wall gate, pin the
private checkout at 2026.10.25.1, and -- until batch 0 has run -- carry provisional batch-1 walls
that refuse to run. The shared machinery (`_common.py`) must leave the 2a and 2b submits
byte-identical. Modules are loaded by path; no cell script is run (the drivers' pure helpers are
checked in test_multi_dataset_datacube_standard.py).
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


p2f = _load(
    "multi_dataset_datacube_p2f", ROOT / "scripts/misc/campaigns/multi_dataset_datacube_p2f.py"
)
p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")
p2b = _load("interferometer_p2b", ROOT / "scripts/misc/campaigns/interferometer_p2b.py")
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from wall import check_submits  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
import catalogue_lifecycle as lc  # noqa: E402

sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

CPU = "hpc/batch_cpu/submit_campaign_p2f_"
GPU = "hpc/batch_gpu/submit_campaign_p2f_"
RAL1 = [f"{CPU}{d}_ral_cpu_{i}_fp64" for d in p2f.DATASET_TYPES for i in p2f.INSTRUMENTS[d]]
A1001 = [f"{GPU}{d}_a100_{i}_fp64" for d in p2f.DATASET_TYPES for i in p2f.INSTRUMENTS[d]]
REFS1 = f"{CPU}references_ral_cpu_fp64"
BATCH1 = RAL1 + A1001 + [REFS1]
BATCH0_RAL = f"{CPU}batch0_ral_cpu_fp64"
BATCH0_A100 = f"{GPU}batch0_a100_fp64"
BATCH0_REFS = f"{CPU}batch0_references_ral_cpu_fp64"
BATCH0 = [BATCH0_RAL, BATCH0_A100, BATCH0_REFS]
ALL = BATCH1 + BATCH0


def planned_cells():
    doc = json.loads((ROOT / "dashboard" / "coverage.json").read_text())
    return {
        c["cell"]
        for c in doc["cells"]
        if c["likelihood"] in ("multi_dataset/delaunay", "datacube/delaunay")
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
    tasks = p2f.planned_tasks()
    assert len(tasks) == len({t.cell for t in tasks}) == 32
    assert {t.cell for t in tasks} == planned_cells()
    assert Counter(t.host_class for t in tasks) == {"hpc_ral_cpu": 18, "hpc_a100": 14}
    assert Counter(t.dataset for t in tasks) == {"multi_dataset": 10, "datacube": 22}
    for t in tasks:
        if t.dataset == "datacube":
            assert t.setup_id.endswith(("_wtilde", "_nufft"))
            if "_dense_" in t.setup_id:  # D1
                assert t.instrument == "sma"
            if t.host_class == "hpc_a100":
                assert t.device == "gpu"
        else:
            assert t.instrument == "hst"
    # Every planned cell's class is the section 3.3 one.
    for t in tasks:
        assert t.setup_id in lc.planned_setups(t.dataset, "delaunay", t.host_class, t.instrument)


def test_d1_scopes_the_dense_datacube_route_to_sma_in_the_catalogue():
    for host in ("hpc_ral_cpu", "hpc_a100"):
        for instrument in ("alma", "alma_high"):
            planned = lc.planned_setups("datacube", "delaunay", host, instrument)
            assert not any("_dense_" in s for s in planned), (host, instrument)
        assert any("_dense_" in s for s in lc.planned_setups("datacube", "delaunay", host, "sma"))


def test_every_task_names_a_real_script_its_flags_parse_and_its_own_cell():
    for task in p2f.planned_tasks():
        assert (ROOT / "scripts" / f"{task.script}.py").is_file(), task.script
        cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
        assert cli.instrument == task.instrument and cli.standard_size
        assert cli.source_pixels is None
        flags = task.flags.split()
        if task.dataset == "datacube":
            assert flags[flags.index("--n-channels") + 1] == "34"
            assert cli.use_sparse_operator == (
                task.backend == "jax"
                and task.operator == "sparse"
                and task.measurement == "runtime"
            )
            assert ("--dense" in flags) == (
                task.backend == "jax"
                and task.operator == "dense"
                and task.measurement == "breakdown"
            )
            if task.backend == "numba":
                assert cli.n_instances == {"sma": 20, "alma": 20, "alma_high": 10}[task.instrument]
        else:
            assert flags[flags.index("--n-datasets") + 1] == "2"
            assert cli.use_sparse_operator == (task.backend == "jax" and task.operator == "sparse")
        assert task.script.endswith("_numba") == (task.backend == "numba")
        assert cli.config_name == p2f.CONFIG_NAMES[(task.host_class, task.device)]
        facts = lc.mirrored(task.output)
        assert facts is not None and facts["location"] == "cell"
        assert (facts["dataset_type"], facts["variant"], facts["instrument"]) == (
            task.dataset,
            "delaunay",
            task.instrument,
        )
        assert (facts["setup_id"], facts["host_class"], facts["file_measurement"]) == (
            task.setup_id,
            task.host_class,
            task.measurement,
        )
        assert facts["model_variant"] == "default"
        assert task.platform == ("cuda,cpu" if task.device == "gpu" else "cpu")


def test_reference_tasks_reuse_2b_cache_paths():
    refs = p2f.reference_tasks()
    assert [t.instrument for t in refs] == ["alma", "alma_high"]
    for t in refs:
        assert t.script == "datacube/delaunay/parity_reference"
        assert "--results-root output/p2f_batch1" in t.flags and "$" not in t.flags
        assert t.output == f"../p2b_references/interferometer_delaunay_{t.instrument}.json"
        assert (ROOT / "scripts" / f"{t.script}.py").is_file()


# -------------------------------------------------------------------------------- submits


def test_submits_are_what_the_generator_renders():
    assert p2f.main(["--check"]) == 0
    assert set(p2f.submits()) == set(ALL)


def test_2a_and_2b_submits_stay_byte_identical_on_the_shared_machinery():
    assert p2a.main(["--check"]) == 0
    assert p2b.main(["--check"]) == 0


def test_batch1_runs_every_cell_once_18_on_ral_and_14_on_gpu():
    ran = Counter()
    for path in RAL1 + A1001:
        text, names = legs(path)
        ran.update(names)
        if path in RAL1:
            assert sbatch(text, "partition") == "ral"
            assert sbatch(text, "array") == f"0-{len(names) - 1}%16"
        else:
            assert sbatch(text, "partition") == "gpu" and "#SBATCH --gres=gpu:1" in text
            assert sbatch(text, "array") is None
    assert set(ran.values()) == {1}
    assert set(ran) == {t.cell for t in p2f.planned_tasks()}
    sizes = {p: len(legs(p)[1]) for p in RAL1 + A1001}
    assert [sizes[p] for p in RAL1] == [4, 6, 4, 4]  # hst, sma, alma, alma_high
    assert [sizes[p] for p in A1001] == [6, 4, 2, 2]
    assert len(legs(REFS1)[1]) == 2


def test_batch0_is_the_bounding_slice():
    _, ral = legs(BATCH0_RAL)
    _, a100 = legs(BATCH0_A100)
    _, refs = legs(BATCH0_REFS)
    assert (len(ral), len(a100), len(refs)) == (10, 10, 1)
    for task in p2f.batch0_ral_cpu() + p2f.batch0_a100():
        if task.dataset == "multi_dataset":
            assert task.instrument == "hst"
        else:
            assert task.instrument == ("sma" if task.operator == "dense" else "alma_high")
    # Every multi_dataset cell is in batch 0 (hst is its only instrument).
    assert {t.cell for t in p2f.batch0_ral_cpu() + p2f.batch0_a100()} >= {
        t.cell for t in p2f.planned_tasks() if t.dataset == "multi_dataset"
    }
    assert {t.instrument for t in p2f.batch0_references()} == {"alma_high"}
    text = (ROOT / BATCH0_A100).read_text()
    assert sbatch(text, "array") == "0-9%2" and "#SBATCH --gres=gpu:1" in text
    assert sbatch((ROOT / BATCH0_RAL).read_text(), "array") == "0-9%8"
    times = {p: sbatch((ROOT / p).read_text(), "time") for p in BATCH0}
    assert times == {BATCH0_RAL: "12:00:00", BATCH0_A100: "6:00:00", BATCH0_REFS: "12:00:00"}
    mems = {p: sbatch((ROOT / p).read_text(), "mem") for p in BATCH0}
    assert mems == {BATCH0_RAL: "128gb", BATCH0_A100: "64gb", BATCH0_REFS: "64gb"}


def test_every_run_writes_to_a_scratch_tree_and_marks_each_leg_start():
    for path in ALL:
        text = (ROOT / path).read_text()
        root = re.search(r'^RESULTS_ROOT="(\S+)"', text, re.M).group(1)
        assert root.startswith("$AP_ROOT/output/p2f_batch"), path
        assert "output/p2f_logs/walls.tsv" in text and "printf 'START" in text
        for out in bash_array(text, "OUTS"):
            assert out.startswith(
                ("results/multi_dataset/", "results/datacube/", "../p2b_references/")
            )


def test_no_cpu_only_job_on_the_gpu_partition():
    for path in ALL:
        text = (ROOT / path).read_text()
        if sbatch(text, "partition") == "gpu":
            assert "#SBATCH --gres=gpu:1" in text, path
        else:
            assert sbatch(text, "partition") == "ral", path
            assert set(bash_array(text, "PLATFORMS")) == {"cpu"}, path


def test_every_submit_passes_the_wall_gate():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert check_submits.check_text(text) == [], path
        rows = check_submits.parse_basis_rows(text)
        cells, _ = check_submits.cells_run(text)
        assert {tuple(r["cell"].split("/")[:2]) for r in rows} == cells, path


def test_batch1_walls_are_measured_by_batch0_and_batch0_never_guards():
    """Every batch-1 row cites batch 0 (RAL 413274, A100 413296, references 413292) on the bounding
    instrument; --time holds 1.25x the slowest leg of a RAL array or the sum of an A100 job's legs,
    plus the slack; --mem holds 1.5x the bounding instrument's peak MaxRSS."""
    assert p2f.WALL_BASIS_STATUS == "measured"
    assert set(p2f.BATCH0_WALLS) == {
        t.cell for t in p2f.batch0_ral_cpu() + p2f.batch0_a100() + p2f.batch0_references()
    }
    assert set(p2f.BATCH0_LIMITATIONS) <= set(p2f.BATCH0_WALLS)
    for path in BATCH1:
        text, names = legs(path)
        assert "WALL_BASIS_STATUS=measured" in text
        assert text.index("WALL_BASIS_STATUS=measured") < text.index(
            'source "$AP_ROOT/activate.sh"'
        )
        tasks = [t for t in p2f.planned_tasks() + p2f.reference_tasks() if t.cell in set(names)]
        assert len(tasks) == len(names), path
        walls = []
        for t in tasks:
            basis = p2f.basis_leg(t)
            assert basis.cell in p2f.BATCH0_WALLS
            assert basis.script == t.script and basis.setup_id == t.setup_id
            wall, elapsed, _ = p2f.BATCH0_WALLS[basis.cell]
            assert p2f.measured_wall(t)[0] == max(wall, elapsed)
            walls.append(max(wall, elapsed))
        need = 1.25 * (sum(walls) if path in A1001 else max(walls)) + p2f.WALL_MIN_SLACK_S
        assert check_submits.parse_slurm_time(sbatch(text, "time")) >= need, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "measured-wall", path
    for path in RAL1 + A1001:
        text, names = legs(path)
        instruments = {t.instrument for t in p2f.planned_tasks() if t.cell in set(names)}
        peak_gib = (
            max(
                p2f.BATCH0_PEAK_RSS_KIB.get(i, max(p2f.BATCH0_PEAK_RSS_KIB.values()))[0]
                for i in instruments
            )
            / 2**20
        )
        mem = int(sbatch(text, "mem").removesuffix("gb"))
        assert mem >= max(32, 1.5 * peak_gib), path
    assert sbatch((ROOT / REFS1).read_text(), "mem") == f"{p2f.REFERENCE_MEM_GB}gb"
    for path in BATCH0:
        text = (ROOT / path).read_text()
        assert "WALL_BASIS_STATUS" not in text and "ALLOW_PROVISIONAL" not in text, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "unmeasured" and float(row["headroom"]) >= 3.0


def test_basis_legs_are_batch0_legs_on_the_bounding_instrument():
    batch0 = {t.cell for t in p2f.batch0_ral_cpu() + p2f.batch0_a100() + p2f.batch0_references()}
    for task in p2f.planned_tasks() + p2f.reference_tasks():
        basis = p2f.basis_leg(task)
        assert basis.cell in batch0, task.cell
        assert basis.script == task.script and basis.setup_id == task.setup_id
        if task.dataset == "datacube":
            order = p2f.INSTRUMENTS["datacube"]
            assert order.index(basis.instrument) >= order.index(task.instrument)


def test_every_submit_pins_the_private_checkout():
    for path in ALL:
        text = (ROOT / path).read_text()
        assert "export P2F_PIN=2026.10.25.1" in text and p2f.PIN == "2026.10.25.1"
        assert "/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1" in text
        assert "REFUSING" in text and "not tag {pin}" in text and "check_revisions" in text
        assert 'AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"' in text
        assert "/mnt/ral/jnightin/PyAuto/PyAuto" not in text
        assert "HPCPullPyAuto" not in text
        assert "from multi_dataset_datacube_p2f import prepare_dataset" in text
        assert "output/numba_cache/p2f_" in text and "output/jax_cache/p2f_" in text
        assert "Decisions D1-D9 adopted" in text


def test_prepare_dataset_dispatches_by_dataset_type(monkeypatch):
    calls = []
    fake_a = type(sys)("imaging_p2a")
    fake_a.prepare_dataset = lambda i, r: calls.append(("imaging", i))
    fake_b = type(sys)("interferometer_p2b")
    fake_b.prepare_dataset = lambda i, r: calls.append(("interferometer", i))
    monkeypatch.setitem(sys.modules, "imaging_p2a", fake_a)
    monkeypatch.setitem(sys.modules, "interferometer_p2b", fake_b)
    for instrument in ("hst", "sma", "alma_high"):
        p2f.prepare_dataset(instrument)
    assert calls == [("imaging", "hst"), ("interferometer", "sma"), ("interferometer", "alma_high")]


def test_mem_measured_bounds_by_the_largest_peak(monkeypatch):
    monkeypatch.setattr(p2f, "BATCH0_PEAK_RSS_KIB", {"alma_high": (40 * 2**20, "x_1")})
    assert p2f.mem_gb_measured(["sma"]) == 64
    assert p2f.mem_gb_measured(["alma_high"]) == 64
    monkeypatch.setattr(p2f, "BATCH0_PEAK_RSS_KIB", {"alma_high": (90 * 2**20, "x_1")})
    assert p2f.mem_gb_measured(["alma"]) == 256
