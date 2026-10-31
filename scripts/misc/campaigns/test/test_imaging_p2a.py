"""The phase-2a imaging campaign manifest and its submits (autolens_profiling#456).

The manifest must be exactly the 106 planned imaging cells; the submits must be what the
generator renders, run every cell once (44 on `ral`, 62 in `--gres=gpu:1` jobs), never put a
CPU-only job on the `gpu` partition, and pass the wall gate. Modules are loaded by path; no cell
script is run.
"""

from __future__ import annotations

import ast
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

RAL1 = [f"hpc/batch_cpu/submit_campaign_p2a_imaging_ral_cpu_{i}_fp64" for i in p2a.INSTRUMENTS]
A1001 = [f"hpc/batch_gpu/submit_campaign_p2a_imaging_a100_{i}_fp64" for i in p2a.INSTRUMENTS]
BATCH1 = RAL1 + A1001
BATCH0 = [
    "hpc/batch_cpu/submit_campaign_p2a_imaging_batch0_ral_cpu_fp64",
    "hpc/batch_gpu/submit_campaign_p2a_imaging_batch0_a100_fp64",
]
RECT28_RAL = "hpc/batch_cpu/submit_campaign_p2a_rect28_ral_cpu_fp64"
RECT28_A100 = "hpc/batch_gpu/submit_campaign_p2a_rect28_a100_fp64"
RECT28 = [RECT28_RAL, RECT28_A100]
JAXRERUN_RAL = [
    f"hpc/batch_cpu/submit_campaign_p2a_jaxrerun_ral_cpu_{i}_fp64" for i in p2a.INSTRUMENTS
]
JAXRERUN_A100 = [
    f"hpc/batch_gpu/submit_campaign_p2a_jaxrerun_a100_{i}_fp64" for i in p2a.INSTRUMENTS
]
JAXRERUN = JAXRERUN_RAL + JAXRERUN_A100


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
    assert_measured_every_designated_cell({t.cell for t in tasks}, planned_imaging_cells())
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
    assert set(p2a.submits()) == set(BATCH1 + BATCH0 + RECT28 + JAXRERUN)


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
    assert_measured_every_designated_cell(seen, planned_imaging_cells())


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
    # Batch 1 requests at least 1.5x batch 0's measured MaxRSS (jwst interpolated, euclid at hst).
    assert {i: p2a.mem_gb_measured([i]) for i in p2a.INSTRUMENTS} == {
        "ao": 64,
        "euclid": 16,
        "hst": 16,
        "jwst": 32,
    }
    for instrument, (kib, _) in p2a.BATCH0_PEAK_RSS_KIB.items():
        assert abs(p2a.peak_rss_gib(instrument) - kib / 2**20) < 1e-9
        assert p2a.mem_gb_measured([instrument]) >= 1.5 * kib / 2**20
    for path in RAL1 + A1001:
        text = (ROOT / path).read_text()
        instrument = path.rsplit("_", 2)[-2]
        floor = p2a.BATCH1_MEM_FLOOR_GB
        expected = {"ao": "64gb", "euclid": "32gb", "hst": "32gb", "jwst": "32gb"}[instrument]
        assert sbatch(text, "mem") == expected
        assert sbatch(text, "mem") == f"{p2a.mem_gb_measured([instrument], minimum=floor)}gb"
        assert "# MEMORY." in text and "MaxRSS" in text
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


def test_batch1_walls_are_measured_by_batch0():
    """README "Setting batch 1's --time": every batch-1 row is measured-wall citing batch 0, and
    --time holds 1.25x the slowest leg of a RAL array or the sum of an A100 job's legs."""
    assert p2a.WALL_BASIS_STATUS == "measured"
    assert set(p2a.BATCH0_WALLS) == {t.cell for t in p2a.batch0_ral_cpu() + p2a.batch0_a100()}
    assert set(p2a.BATCH0_LIMITATIONS) <= set(p2a.BATCH0_WALLS)
    for path in BATCH1:
        text, names = legs(path)
        assert "WALL_BASIS_STATUS=measured" in text
        guard = text.index("WALL_BASIS_STATUS=measured")
        assert guard < text.index('source "$AP_ROOT/activate.sh"')
        tasks = [t for t in p2a.planned_tasks() if t.cell in set(names)]
        walls = []
        for t in tasks:
            basis = p2a.basis_leg(t)
            assert basis.cell in p2a.BATCH0_WALLS  # a leg of the same script and setup
            assert basis.script == t.script and basis.instrument in ("ao", "hst")
            wall, elapsed, _ = p2a.BATCH0_WALLS[basis.cell]
            assert p2a.measured_wall(t)[0] == max(wall, elapsed)
            walls.append(max(wall, elapsed))
        need = 1.25 * (max(walls) if path in RAL1 else sum(walls)) + p2a.WALL_MIN_SLACK_S
        assert check_submits.parse_slurm_time(sbatch(text, "time")) >= need, path
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "measured-wall" and float(row["headroom"]) >= 1.25
            refs = row["ref"].split(",")
            assert all(r.split("_")[0] in {"412877", "412884", "412911", "412915"} for r in refs)


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


def _module_assignments(tree, name):
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]


def test_every_script_a_leg_runs_honours_instrument_and_standard_size():
    """For every distinct (script, flags) the generator emits, read that script's AST: the
    instrument comes from --instrument (one module-level binding, never reassigned), the dataset
    path is built from it, and a pixelized leg's script builds its size from --standard-size."""
    seen = {}
    for task in p2a.planned_tasks():
        seen.setdefault(task.script, set()).add(task.variant)
    assert len(seen) == 12
    for script, variants in sorted(seen.items()):
        source = (ROOT / "scripts" / f"{script}.py").read_text()
        tree = ast.parse(source)
        [binding] = _module_assignments(tree, "instrument")
        assert ast.unparse(binding.value).startswith("_cli.instrument or "), script
        rebinds = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and n is not binding
            and any(isinstance(t, ast.Name) and t.id == "instrument" for t in n.targets)
        ]
        assert not rebinds, script
        assert 'Path("dataset") / "imaging" / instrument' in source, script
        assert '"dataset/imaging/hst' not in source and "'dataset/imaging/hst" not in source
        if variants & set(p2a.PIXELIZED):
            if script.endswith("_numba"):
                assert "standard_size_preset(" in source and "preset.production_data" in source
            else:
                for variant in variants:
                    assert f'standard_jax_cell(\n        "{variant}",' in source, script
        # the leg's flags reach the CLI the script parses
        for task in (t for t in p2a.planned_tasks() if t.script == script):
            cli = pc.parse_profile_cli(argv=["--instrument", task.instrument, *task.flags.split()])
            assert cli.instrument == task.instrument


# ---------------------------------------------------- 28x28 rectangular size variant (#468)


def test_rect28_cells_are_the_inference_setups_at_784_in_their_size_folder():
    tasks = p2a.rect28_tasks()
    assert [t.cell for t in tasks] == sorted(
        f"imaging/rectangular/hst {sid}@{host} [{m}] source_pixels_784"
        for sid, host in (
            ("numba_cpu_sparse_fp64", "hpc_ral_cpu"),
            ("jax_cpu_sparse_fp64", "hpc_a100"),
            ("jax_gpu_sparse_fp64", "hpc_a100"),
            ("jax_gpu_dense_fp64", "hpc_a100"),
        )
        for m in ("breakdown", "runtime")
    )
    assert "784" in json.dumps(json.loads((ROOT / lc.SIZE_VARIANTS).read_text()))
    for task in tasks:
        # The 2a standard cell of the same setup, plus the declared size: the same preset.
        standard = p2a.Task(
            *[getattr(task, f) for f in ("variant", "instrument", "setup_id")],
            task.host_class,
            task.measurement,
        )
        assert standard in p2a.planned_tasks() and task.script == standard.script
        assert task.flags == standard.flags.replace(
            "--standard-size", "--standard-size --source-pixels 784"
        )
        cli = pc.parse_profile_cli(argv=["--instrument", "hst", *task.flags.split()])
        assert cli.standard_size and cli.source_pixels == 784
        assert pc.requested_source_pixels(cli, "rectangular") == 784
        facts = lc.mirrored(task.output)
        assert (facts["location"], facts["model_variant"]) == ("cell", "default")
        assert facts["size_variant"] == "source_pixels_784"
        assert (facts["setup_id"], facts["host_class"], facts["file_measurement"]) == (
            task.setup_id,
            task.host_class,
            task.measurement,
        )
        assert p2a.basis_leg(task) == p2a.basis_leg(standard)  # the standard's measured wall


def test_rect28_preset_is_the_standard_production_cell_at_28x28():
    import _production_config as prod

    standard = prod.standard_size_preset("pixelization_numba", "hst")
    sized = prod.standard_size_preset("pixelization_numba", "hst", 784)
    assert (standard.rect_pixels_yx, sized.rect_pixels_yx) == (35, 28)
    assert prod.standard_size_preset("pixelization_numba", "hst", 1225) == standard
    differs = {
        f for f in standard.__dataclass_fields__ if getattr(standard, f) != getattr(sized, f)
    }
    assert differs == {"rect_pixels_yx", "name", "provenance", "size_override"}
    assert sized.production_data and "declared size variant" in sized.size_override


def test_rect28_flags_outside_the_declared_variants_are_refused():
    import pytest

    with pytest.raises(SystemExit):
        pc.parse_profile_cli(argv=["--standard-size", "--source-pixels", "1250"])
    cli = pc.parse_profile_cli(argv=["--standard-size", "--source-pixels", "784"])
    with pytest.raises(ValueError, match="not a declared size variant"):
        pc.requested_source_pixels(cli, "delaunay")


def test_rect28_submits_partition_pin_scratch_tree_and_legs():
    ral, ral_names = legs(RECT28_RAL)
    gpu, gpu_names = legs(RECT28_A100)
    assert ral_names == [t.cell for t in p2a.rect28_tasks("hpc_ral_cpu")]
    assert gpu_names == [t.cell for t in p2a.rect28_tasks("hpc_a100")]
    assert len(ral_names) == 2 and len(gpu_names) == 6
    assert sbatch(ral, "partition") == "ral" and "--gres" not in ral
    assert sbatch(ral, "array") == "0-1%16"
    assert sbatch(gpu, "partition") == "gpu" and sbatch(gpu, "gres") == "gpu:1"
    assert sbatch(gpu, "array") is None  # one job, legs in turn
    platforms = dict(zip(gpu_names, bash_array(gpu, "PLATFORMS"), strict=True))
    for name, platform in platforms.items():
        assert platform == ("cpu" if "jax_cpu_sparse" in name else "cuda,cpu")
    for text, tasks in (
        (ral, p2a.rect28_tasks("hpc_ral_cpu")),
        (gpu, p2a.rect28_tasks("hpc_a100")),
    ):
        assert bash_array(text, "FLAGS") == [t.flags for t in tasks]
        assert bash_array(text, "OUTS") == [t.output for t in tasks]
        assert all("/source_pixels_784/" in o for o in bash_array(text, "OUTS"))
        assert all("--standard-size --source-pixels 784" in f for f in bash_array(text, "FLAGS"))
        assert p2a.RECT28_PIN == "2026.10.25.1"
        assert "export P2A_PIN=2026.10.25.1" in text and "2026.10.23.1" not in text
        assert (
            'PYAUTO_PRIVATE_BASE="${PYAUTO_PRIVATE_BASE:-'
            '/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1}"' in text
        )
        assert "REFUSING" in text and "check_revisions" in text and "describe --tags" in text
        assert 'RESULTS_ROOT="$AP_ROOT/output/p2a_rect28"' in text
        assert "HPCPullPyAuto" not in text and "/mnt/ral/jnightin/PyAuto/PyAuto" not in text
        assert check_submits.check_text(text) == []
        assert sbatch(text, "mem") == f"{p2a.mem_gb_measured(['hst'], minimum=32)}gb" == "32gb"
        for row in check_submits.parse_basis_rows(text):
            assert row["source"] == "measured-wall" and row["measured-on"] == "ao"
        assert "upper bounds" in text


def test_rect28_time_holds_the_standard_walls():
    for path, sequential in ((RECT28_RAL, False), (RECT28_A100, True)):
        text, names = legs(path)
        tasks = [t for t in p2a.rect28_tasks() if t.cell in set(names)]
        walls = [p2a.measured_wall(t)[0] for t in tasks]
        need = 1.25 * (sum(walls) if sequential else max(walls)) + p2a.WALL_MIN_SLACK_S
        assert check_submits.parse_slurm_time(sbatch(text, "time")) >= need, path
    assert sbatch((ROOT / RECT28_RAL).read_text(), "time") == "1:00:00"
    assert sbatch((ROOT / RECT28_A100).read_text(), "time") == "1:30:00"


def test_jax_rerun_submits_are_every_jax_batch1_leg_on_the_2026_10_25_1_pin():
    """Phase 3, Q7 (human 2026-10-30): every JAX leg of the batch-1 submits, and only those, rerun
    on tag 2026.10.25.1 into their own scratch tree; numba rows are carried."""
    every = []
    for instrument, ral_path, gpu_path in zip(p2a.INSTRUMENTS, JAXRERUN_RAL, JAXRERUN_A100):
        for path, host_class, batch1 in (
            (ral_path, "hpc_ral_cpu", p2a.batch1_ral_cpu(instrument)),
            (gpu_path, "hpc_a100", p2a.batch1_a100(instrument)),
        ):
            text, names = legs(path)
            tasks = p2a.jax_rerun_tasks(instrument, host_class)
            assert names == [t.cell for t in tasks]
            assert names == [t.cell for t in batch1 if t.setup_id.startswith("jax_")]
            assert bash_array(text, "FLAGS") == [t.flags for t in tasks]
            assert "export P2A_PIN=2026.10.25.1" in text and "2026.10.23.1" not in text
            assert 'RESULTS_ROOT="$AP_ROOT/output/p3_jax_rerun"' in text
            assert check_submits.check_text(text) == []
            if host_class == "hpc_ral_cpu":
                assert sbatch(text, "partition") == "ral" and "--gres" not in text
            else:
                assert sbatch(text, "partition") == "gpu" and sbatch(text, "gres") == "gpu:1"
            every += names
    assert len(every) == len(set(every)) == 88
    assert all("numba" not in n for n in every)
