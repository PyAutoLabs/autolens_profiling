"""The phase-2b interferometer campaign manifest and its submits (autolens_profiling#469).

The manifest must be exactly the 60 planned interferometer cells (rulings R1(a), R2, R9); the
submits must be what the generator renders, run every cell once (38 on `ral`, 22 in
`--gres=gpu:1` jobs), never put a CPU-only job on the `gpu` partition, pass the wall gate, pin
the private checkout at 2026.10.25.1 and cite batch 0's measured walls and MaxRSS in batch 1. The
shared machinery (`_common.py`) must leave the phase-2a submits byte-identical. Modules are
loaded by path; no cell script is run.
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


p2b = _load("interferometer_p2b", ROOT / "scripts/misc/campaigns/interferometer_p2b.py")
p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from wall import check_submits  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
import catalogue_lifecycle as lc  # noqa: E402

sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

CPU = "hpc/batch_cpu/submit_campaign_p2b_"
GPU = "hpc/batch_gpu/submit_campaign_p2b_"
RAL1 = [f"{CPU}interferometer_ral_cpu_{i}_fp64" for i in p2b.INSTRUMENTS]
A1001 = [f"{GPU}interferometer_a100_{i}_fp64" for i in p2b.INSTRUMENTS]
REFS1 = f"{CPU}references_ral_cpu_fp64"
BATCH1 = RAL1 + A1001 + [REFS1]
BATCH0_RAL = f"{CPU}batch0_ral_cpu_fp64"
BATCH0_A100 = f"{GPU}batch0_a100_fp64"
BATCH0_REFS = f"{CPU}batch0_references_ral_cpu_fp64"
BATCH0 = [BATCH0_RAL, BATCH0_A100, BATCH0_REFS]
MATRIXFREE = f"{CPU}matrixfree_ral_cpu_fp64"
ALL = BATCH1 + BATCH0 + [MATRIXFREE]


def planned_interferometer_cells():
    doc = json.loads((ROOT / "dashboard" / "coverage.json").read_text())
    return {c["cell"] for c in doc["cells"] if c["likelihood"].startswith("interferometer/")}


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


def test_manifest_is_exactly_the_planned_interferometer_cells():
    tasks = p2b.planned_tasks()
    assert len(tasks) == len({t.cell for t in tasks}) == 60
    assert {t.cell for t in tasks} == planned_interferometer_cells()
    assert Counter(t.host_class for t in tasks) == {"hpc_ral_cpu": 38, "hpc_a100": 22}
    # R1(a): dense pixelized NUFFT only at sma; R2: no DFT cell; no jax_cpu leg on the A100.
    for t in tasks:
        assert not t.setup_id.endswith("_dft")
        if "_dense_" in t.setup_id:
            assert t.instrument == "sma"
        if t.host_class == "hpc_a100":
            assert t.device == "gpu"
    assert {t.instrument for t in tasks if t.variant == "mge"} == {"sma"}


def test_every_task_names_a_real_script_its_flags_parse_and_its_own_cell():
    for task in p2b.planned_tasks():
        assert (ROOT / "scripts" / f"{task.script}.py").is_file(), task.script
        cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
        assert cli.instrument == task.instrument
        assert cli.standard_size == (task.variant in p2b.PIXELIZED)
        assert cli.use_sparse_operator == (
            task.backend == "jax" and task.operator == "sparse" and task.measurement == "runtime"
        )
        assert cli.source_pixels is None
        assert ("--dense" in task.flags.split()) == (
            task.backend == "jax"
            and task.operator == "dense"
            and task.measurement == "breakdown"
            and task.variant in p2b.PIXELIZED
        )
        if task.backend == "numba":
            assert (
                cli.n_instances
                == {"sma": 20, "alma": 20, "alma_high": 10, "jvla": 5}[task.instrument]
            )
            assert task.script.endswith("_numba")
        else:
            assert not task.script.endswith("_numba")
        assert (
            cli.config_name
            == {"hpc_ral_cpu": "hpc_ral_cpu_fp64", "hpc_a100": "hpc_a100_fp64"}[task.host_class]
        )
        facts = lc.mirrored(task.output)
        assert facts is not None and facts["location"] == "cell"
        assert (facts["variant"], facts["instrument"], facts["setup_id"]) == (
            task.variant,
            task.instrument,
            task.setup_id,
        )
        assert (facts["host_class"], facts["file_measurement"]) == (
            task.host_class,
            task.measurement,
        )
        assert facts["model_variant"] == "default"
        assert task.platform == ("cuda,cpu" if task.device == "gpu" else "cpu")


def test_reference_and_matrixfree_tasks():
    refs = p2b.reference_tasks()
    assert {(t.variant, t.instrument) for t in refs} == {
        (v, i) for v in ("rectangular", "delaunay") for i in ("alma", "alma_high", "jvla")
    }
    for t in refs:
        assert t.script == "interferometer/pixelized/parity_reference"
        assert f"--likelihood {t.variant}" in t.flags and "$" not in t.flags
        assert t.output == f"../p2b_references/interferometer_{t.variant}_{t.instrument}.json"
    mf = p2b.matrixfree_tasks()
    assert [(t.instrument, t.mesh_side) for t in mf] == [
        ("sma", 35),
        ("alma", 35),
        ("alma_high", 35),
        ("jvla", 35),
        ("alma", 64),
        ("alma", 128),
        ("alma", 192),
    ]
    for t in mf:
        facts = lc.mirrored(t.output)
        assert facts is not None and facts["location"] in ("experiments", "scans")
        assert facts["setup_id"] == "numba_cpu_sparse_fp64_wtilde"
        assert facts["location"] == ("experiments" if t.mesh_side == 35 else "scans")
        assert (ROOT / "scripts" / f"{t.script}.py").is_file()


def test_every_batch1_limitation_names_a_leg_with_no_committed_row():
    legs = {t.cell: t for t in p2b.planned_tasks() + p2b.matrixfree_tasks()}
    assert set(p2b.BATCH1_LIMITATIONS) <= set(legs)
    for cell, reason in p2b.BATCH1_LIMITATIONS.items():
        assert reason and not (ROOT / legs[cell].output).exists()


# -------------------------------------------------------------------------------- submits


def test_submits_are_what_the_generator_renders():
    assert p2b.main(["--check"]) == 0
    assert set(p2b.submits()) == set(ALL)


def test_phase_2a_submits_stay_byte_identical_on_the_shared_machinery():
    assert p2a.main(["--check"]) == 0
    for rel, text in p2a.submits().items():
        assert (ROOT / rel).read_text() == text, rel


def test_batch1_runs_every_cell_once_38_on_ral_and_22_on_gpu():
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
    assert set(ran) == {t.cell for t in p2b.planned_tasks()}
    sizes = {p: len(legs(p)[1]) for p in RAL1 + A1001}
    assert [sizes[p] for p in RAL1] == [14, 8, 8, 8]
    assert [sizes[p] for p in A1001] == [10, 4, 4, 4]
    assert len(legs(REFS1)[1]) == 6


def test_batch0_is_the_bounding_slice():
    _, ral = legs(BATCH0_RAL)
    _, a100 = legs(BATCH0_A100)
    _, refs = legs(BATCH0_REFS)
    assert (len(ral), len(a100), len(refs)) == (14, 10, 2)
    for task in p2b.batch0_ral_cpu() + p2b.batch0_a100():
        dense = task.variant == "mge" or task.operator == "dense"
        assert task.instrument == ("sma" if dense else "jvla")
    assert Counter(t.setup_id for t in p2b.batch0_ral_cpu()) == {
        "numba_cpu_sparse_fp64_wtilde": 4,
        "jax_cpu_sparse_fp64_wtilde": 4,
        "jax_cpu_dense_fp64_nufft": 6,
    }
    assert {t.instrument for t in p2b.batch0_references()} == {"jvla"}
    text = (ROOT / BATCH0_A100).read_text()
    assert sbatch(text, "array") == "0-9%2" and "#SBATCH --gres=gpu:1" in text
    assert sbatch((ROOT / BATCH0_RAL).read_text(), "array") == "0-13%8"
    times = {p: sbatch((ROOT / p).read_text(), "time") for p in BATCH0}
    assert times == {BATCH0_RAL: "12:00:00", BATCH0_A100: "6:00:00", BATCH0_REFS: "24:00:00"}
    mems = {p: sbatch((ROOT / p).read_text(), "mem") for p in BATCH0}
    assert mems == {BATCH0_RAL: "128gb", BATCH0_A100: "64gb", BATCH0_REFS: "64gb"}


def test_every_run_writes_to_a_scratch_tree_and_marks_each_leg_start():
    for path in ALL:
        text = (ROOT / path).read_text()
        root = re.search(r'^RESULTS_ROOT="(\S+)"', text, re.M).group(1)
        assert root.startswith("$AP_ROOT/output/p2b_"), path
        assert "output/p2b_logs/walls.tsv" in text and "printf 'START" in text
        for out in bash_array(text, "OUTS"):
            assert out.startswith("results/interferometer/") or out.startswith("../p2b_references/")
    assert '"$AP_ROOT/output/p2b_matrixfree"' in (ROOT / MATRIXFREE).read_text()


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


def test_batch1_walls_are_measured_by_batch0_and_batch0_never_guards():
    """Every batch-1 row is measured-wall citing batch 0 (RAL 413115, references 413116, A100
    413117), and --time holds 1.25x the slowest leg of a RAL array or the sum of an A100 job's
    legs, plus the slack."""
    assert p2b.WALL_BASIS_STATUS == "measured"
    batch0 = p2b.batch0_ral_cpu() + p2b.batch0_a100() + p2b.batch0_references()
    assert set(p2b.BATCH0_WALLS) == {t.cell for t in batch0}
    assert set(p2b.BATCH0_LIMITATIONS) <= set(p2b.BATCH0_WALLS)
    for path in BATCH1:
        text, names = legs(path)
        assert "WALL_BASIS_STATUS=measured" in text
        assert text.index("WALL_BASIS_STATUS=measured") < text.index(
            'source "$AP_ROOT/activate.sh"'
        )
        tasks = [t for t in p2b.planned_tasks() + p2b.reference_tasks() if t.cell in set(names)]
        assert len(tasks) == len(names)
        walls = []
        for t in tasks:
            basis = p2b.basis_leg(t)
            assert basis.cell in p2b.BATCH0_WALLS  # a leg of the same script and setup
            assert basis.script == t.script and basis.setup_id == t.setup_id
            # Never bounded by a smaller instrument: jvla bounds every sparse / reference leg.
            assert p2b.INSTRUMENTS.index(basis.instrument) >= p2b.INSTRUMENTS.index(t.instrument)
            wall, elapsed, _ = p2b.BATCH0_WALLS[basis.cell]
            assert p2b.measured_wall(t)[0] == max(wall, elapsed)
            walls.append(max(wall, elapsed))
        need = 1.25 * (sum(walls) if path in A1001 else max(walls)) + p2b.WALL_MIN_SLACK_S
        assert check_submits.parse_slurm_time(sbatch(text, "time")) >= need, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "measured-wall" and float(row["headroom"]) >= 1.25
            assert all(
                r.split("_")[0] in {"413115", "413116", "413117"} for r in row["ref"].split(",")
            )
    times = {p: sbatch((ROOT / p).read_text(), "time") for p in RAL1 + A1001 + [REFS1]}
    assert {times[p] for p in RAL1} == {"2:30:00"}
    assert {times[p] for p in A1001} == {"1:30:00"}
    assert times[REFS1] == "4:00:00"
    for path in BATCH0 + [MATRIXFREE]:
        text = (ROOT / path).read_text()
        assert "WALL_BASIS_STATUS" not in text and "ALLOW_PROVISIONAL" not in text, path


def test_batch1_mem_is_batch0_maxrss_times_headroom():
    for table in (p2b.BATCH0_PEAK_RSS_KIB, p2b.REFERENCE_PEAK_RSS_KIB):
        for instrument, (kib, ref) in table.items():
            assert abs(p2b.peak_rss_gib(instrument, table) - kib / 2**20) < 1e-9
            assert p2b.mem_gb_measured([instrument], table=table) >= 1.5 * kib / 2**20
            assert ref.split("_")[0] in {"413115", "413116", "413117"}
    # alma and alma_high are bounded by jvla's peak, never by sma's.
    assert p2b.peak_rss_gib("alma") == p2b.peak_rss_gib("jvla")
    assert p2b.peak_rss_gib("alma_high") == p2b.peak_rss_gib("jvla")
    for path in RAL1 + A1001 + [REFS1]:
        text = (ROOT / path).read_text()
        assert sbatch(text, "mem") == "64gb", path
        assert "# MEMORY." in text and "MaxRSS" in text


def test_every_submit_pins_the_private_checkout():
    for path in BATCH1 + BATCH0:
        text = (ROOT / path).read_text()
        assert "export P2B_PIN=2026.10.25.1" in text and p2b.PIN == "2026.10.25.1"
        assert "/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1" in text
        assert "REFUSING" in text and "not tag {pin}" in text and "check_revisions" in text
    for path in ALL:
        text = (ROOT / path).read_text()
        assert 'AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"' in text
        assert "/mnt/ral/jnightin/PyAuto/PyAuto" not in text
        assert "HPCPullPyAuto" not in text
        assert "from interferometer_p2b import prepare_dataset" in text
        assert "output/numba_cache/p2b_" in text and "output/jax_cache/p2b_" in text


def test_matrixfree_submit_guards_on_the_declared_main_shas():
    text = (ROOT / MATRIXFREE).read_text()
    assert 'PYAUTO_PRIVATE_BASE="${MF_PRIVATE_BASE:-' in text
    assert "declared main SHA" in text and "not tag {pin}" not in text
    for repo, sha in p2b.MF_LIBRARY_SHAS.items():
        assert f'"{repo}": "{sha}"' in text
    assert "check_revisions" in text and "UNRELEASED" in text
    _, names = legs(MATRIXFREE)
    assert len(names) == 7 and sbatch(text, "partition") == "ral"
    for out in bash_array(text, "OUTS"):
        assert "/experiments/numpy_cpu_matrixfree/" in out or "/scans/numpy_cpu_matrixfree_" in out


def test_mem_measured_bounds_by_the_next_measured_instrument():
    table = {"sma": (10 * 2**20, "x_1"), "jvla": (60 * 2**20, "x_2")}
    assert p2b.peak_rss_gib("sma", table) == 10
    assert p2b.peak_rss_gib("alma", table) == 60
    assert p2b.mem_gb_measured(["sma"], table=table) == 32
    assert p2b.mem_gb_measured(["jvla"], table=table) == 128
