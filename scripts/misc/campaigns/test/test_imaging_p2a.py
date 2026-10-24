"""The phase-2a imaging campaign manifest and its submits (autolens_profiling#456).

The manifest must be exactly the 106 planned imaging cells; the submits must be what the
generator renders, run every cell once (44 on `ral`, 62 in `--gres=gpu:1` jobs), never put a
CPU-only job on the `gpu` partition, and pass the wall gate. Modules are loaded by path; no cell
script is run.
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


p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from wall import check_submits  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
import catalogue_lifecycle as lc  # noqa: E402

sys.path.insert(0, str(ROOT))
import _profile_cli as pc  # noqa: E402

RAL1 = [f"hpc/batch_cpu/submit_campaign_p2a_imaging_ral_cpu_{i}_fp64" for i in p2a.INSTRUMENTS]
A1001 = [f"hpc/batch_gpu/submit_campaign_p2a_imaging_a100_{i}_fp64" for i in p2a.INSTRUMENTS]
BATCH1 = RAL1 + A1001
BATCH0 = [
    "hpc/batch_cpu/submit_campaign_p2a_imaging_batch0_ral_cpu_fp64",
    "hpc/batch_gpu/submit_campaign_p2a_imaging_batch0_a100_fp64",
]


def planned_imaging_cells():
    doc = json.loads((ROOT / "dashboard" / "coverage.json").read_text())
    return {c["cell"] for c in doc["cells"] if c["likelihood"].startswith("imaging/")}


def bash_array(text, name):
    match = re.search(rf"^{name}=\(\n(.*?)\n\)", text, re.M | re.S)
    assert match, name
    return [shlex.split(line)[0] if line.strip() else "" for line in match.group(1).splitlines()]


def sbatch(text, key):
    match = re.search(rf"^#SBATCH --{key}=(\S+)", text, re.M)
    return match.group(1) if match else None


# ------------------------------------------------------------------------------- manifest


def test_manifest_is_exactly_the_planned_imaging_cells():
    tasks = p2a.planned_tasks()
    assert len(tasks) == len({t.cell for t in tasks}) == 106
    assert {t.cell for t in tasks} == planned_imaging_cells()
    hosts = Counter(t.host_class for t in tasks)
    assert hosts == {"hpc_ral_cpu": 44, "hpc_a100": 62}
    a100_cpu = [t for t in tasks if t.host_class == "hpc_a100" and t.device == "cpu"]
    assert len(a100_cpu) == 18 and {t.setup_id for t in a100_cpu} == {"jax_cpu_sparse_fp64"}


def test_every_task_names_a_real_script_flags_its_cli_accepts_and_its_own_cell():
    for task in p2a.planned_tasks():
        assert (ROOT / "scripts" / f"{task.script}.py").is_file(), task.script
        cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
        assert cli.standard_size == (task.variant in p2a.PIXELIZED)
        assert cli.use_sparse_operator == (task.backend == "jax" and task.operator == "sparse")
        assert cli.source_pixels is None  # never the --source-pixels scan path
        assert ("--mesh-class delaunay_nn" in task.flags) == (
            task.backend == "numba" and task.variant == "delaunay_nn"
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


def test_numba_and_jax_scripts_by_setup():
    for task in p2a.planned_tasks():
        assert task.script.endswith("_numba") == (task.backend == "numba")
        if task.variant == "delaunay_nn" and task.backend == "numba":
            assert task.script == f"imaging/delaunay/likelihood_{task.measurement}_numba"


# -------------------------------------------------------------------------------- submits


def test_submits_are_what_the_generator_renders():
    assert p2a.main(["--check"]) == 0
    assert set(p2a.submits()) == set(BATCH1 + BATCH0)


def legs(path):
    text = (ROOT / path).read_text()
    names = bash_array(text, "CELLNAMES")
    columns = {k: bash_array(text, k) for k in ("CELLS", "INSTS", "PLATFORMS", "OUTS")}
    assert all(len(v) == len(names) for v in columns.values())
    return text, names


def test_batch1_runs_every_cell_once_44_on_ral_and_62_on_gpu():
    seen, ral = [], 0
    for path in BATCH1:
        text, names = legs(path)
        seen += names
        instrument = path.rsplit("_", 2)[-2]
        assert set(bash_array(text, "INSTS")) == {instrument}  # one instrument per submit
        if "batch_cpu" in path:
            assert sbatch(text, "partition") == "ral" and "--gres" not in text
            assert sbatch(text, "array") == f"0-{len(names) - 1}%16"
            assert all("@hpc_ral_cpu" in n for n in names)
            ral += len(names)
        else:
            assert sbatch(text, "partition") == "gpu" and sbatch(text, "gres") == "gpu:1"
            assert sbatch(text, "array") is None  # one job per instrument, legs in turn
            assert all("@hpc_a100" in n for n in names)
    assert len(seen) == len(set(seen)) == 106 and ral == 44
    assert set(seen) == planned_imaging_cells()


def test_memory_follows_the_dataset_size():
    import math

    sys.path.insert(0, str(ROOT / "scripts" / "misc"))
    from simulators.imaging import INSTRUMENTS as SIMULATED

    for instrument, scale in p2a.PIXEL_SCALES.items():
        assert SIMULATED[instrument]["pixel_scale"] == scale
        assert p2a.masked_pixels(instrument) == int(math.pi * 3.5**2 / scale**2)
    assert {i: p2a.mem_gb([i]) for i in p2a.INSTRUMENTS} == {
        "ao": 128,
        "euclid": 16,
        "hst": 16,
        "jwst": 16,
    }
    for path in RAL1 + A1001:
        text = (ROOT / path).read_text()
        instrument = path.rsplit("_", 2)[-2]
        floor = 16 if path in RAL1 else 32
        assert sbatch(text, "mem") == f"{p2a.mem_gb([instrument], minimum=floor)}gb"
        assert "# MEMORY." in text
    for path in BATCH0:  # the bounding slice holds ao
        assert sbatch((ROOT / path).read_text(), "mem") == "128gb"


def test_every_run_writes_to_a_scratch_tree_and_marks_each_leg_start():
    for batch, paths in ((1, BATCH1), (0, BATCH0)):
        for path in paths:
            text = (ROOT / path).read_text()
            assert f'RESULTS_ROOT="$AP_ROOT/output/p2a_batch{batch}"' in text
            assert 'RESULTS_ROOT="$AP_ROOT"' not in text  # never in place over results/
            assert text.index("printf 'START") < text.index("python3 -u scripts/")
            assert text.index("python3 -u scripts/") < text.index("printf 'WALL")


def test_no_cpu_only_job_on_the_gpu_partition():
    for path in check_submits.submit_paths(ROOT):
        if not path.name.startswith("submit_campaign_p2a_"):
            continue
        text = path.read_text()
        if sbatch(text, "partition") != "ral":
            assert sbatch(text, "gres") == "gpu:1", path.name
        assert "ral,gpu" not in text and "gpu,ral" not in text


def test_a100_cpu_legs_run_inside_the_gpu_jobs_on_the_cpu_platform():
    for path in A1001:
        text, names = legs(path)
        platforms = bash_array(text, "PLATFORMS")
        for name, platform in zip(names, platforms, strict=True):
            assert platform == ("cpu" if "jax_cpu_sparse" in name else "cuda,cpu")


def test_every_submit_passes_the_wall_gate_with_a_row_per_cell_and_instrument():
    for path in BATCH1 + BATCH0:
        text = (ROOT / path).read_text()
        assert check_submits.check_text(text) == [], path
        rows = check_submits.parse_basis_rows(text)
        run_cells, instruments = check_submits.cells_run(text)
        declared = {tuple(r["cell"].split("/")) for r in rows}
        expected = {
            (*t.gate_cell.split("/"), t.instrument)
            for t in p2a.planned_tasks()
            if t.cell in set(bash_array(text, "CELLNAMES"))
        }
        assert declared == expected, path
        assert {(d, c) for d, c, _ in declared} == run_cells
        assert {i for *_, i in declared} == instruments


def test_batch1_is_guarded_until_batch0_measures_its_walls():
    for path in BATCH1:
        text = (ROOT / path).read_text()
        assert "WALL_BASIS_STATUS=provisional" in text
        assert all(r["source"] == "unmeasured" for r in check_submits.parse_basis_rows(text))
        guard = text.index("WALL_BASIS_STATUS=provisional")
        assert guard < text.index('source "$AP_ROOT/activate.sh"')


def test_batch0_is_the_bounding_slice_and_writes_outside_results():
    expected = {
        t.cell
        for t in p2a.planned_tasks()
        if (t.instrument == "ao" and t.variant != "delaunay_nn")
        or (t.instrument == "hst" and t.variant == "delaunay_nn")
    }
    seen = []
    for path in BATCH0:
        text, names = legs(path)
        seen += names
        assert 'RESULTS_ROOT="$AP_ROOT/output/p2a_batch0"' in text
        assert "WALL_BASIS_STATUS" not in text
        allowed = check_submits.parse_slurm_time(sbatch(text, "time"))
        for row in check_submits.parse_basis_rows(text):
            # One leg per array task: the guess plus the headline median's extra calls, 3x.
            need = 3.0 * (float(row["wall"]) + check_submits.headline_median_extra_wall_s())
            assert allowed >= need
        if "batch_gpu" in path:
            assert sbatch(text, "array").endswith("%2") and sbatch(text, "gres") == "gpu:1"
        else:
            assert sbatch(text, "partition") == "ral"
    assert sorted(seen) == sorted(expected) and len(seen) == 34


def test_every_submit_pins_the_private_checkout_at_the_release_tag():
    for path in BATCH1 + BATCH0:
        text = (ROOT / path).read_text()
        assert f"export P2A_PIN={p2a.PIN}" in text and p2a.PIN == "2026.10.23.1"
        assert 'AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"' in text
        assert "PYAUTO_HPC_BASE=" in text and "PYAUTO_PRIVATE_BASE=" in text
        assert "REFUSING" in text and "check_revisions" in text
        assert "/mnt/ral/jnightin/PyAuto/PyAuto" not in text
        assert "HPCPullPyAuto" not in text
        assert "--standard-size" in text
