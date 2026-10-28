"""Likelihood-lifecycle phase 2f: the multi_dataset + datacube campaign's tasks and SLURM submits.

autolens_profiling#475 measures the planned ``multi_dataset/delaunay`` and ``datacube/delaunay``
cells (``dashboard/coverage.json``) at the standard size on the pinned release 2026.10.25.1 (as
2b), from a private RAL checkout. This module is the one copy of that work list: every cell
becomes one :class:`Task` (script, flags, output path) and the submits under ``hpc/`` are rendered
from the tasks with the machinery 2a and 2b share (``_common.py``)::

    python scripts/misc/campaigns/multi_dataset_datacube_p2f.py --write   # (re)render the submits
    python scripts/misc/campaigns/multi_dataset_datacube_p2f.py --check   # exit 1 if one is stale
    python scripts/misc/campaigns/multi_dataset_datacube_p2f.py --list    # print the manifest

DRAFT FOR REVIEW (the human, 2026-10-28): nothing here has been submitted. The decisions D1-D9
(wiki/campaigns/datacube/likelihood_lifecycle_p2f.md) await the human before batch 0 runs.

The submits (README.md beside this file has the batch procedure):

- batch 0, the timing arm (decision D9): ``hpc/batch_cpu/submit_campaign_p2f_batch0_ral_cpu_fp64``
  (``--partition=ral`` array of 10), ``hpc/batch_gpu/submit_campaign_p2f_batch0_a100_fp64``
  (``--gres=gpu:1`` array of 10, ``%2``) and the optional alma_high reference
  ``hpc/batch_cpu/submit_campaign_p2f_batch0_references_ral_cpu_fp64`` (1), into the gitignored
  ``output/p2f_batch0/``;
- batch 1, one ``--partition=ral`` array per (dataset type, instrument)
  (``submit_campaign_p2f_<dataset>_ral_cpu_<instrument>_fp64``), one ``--gres=gpu:1`` job per
  (dataset type, instrument) (``submit_campaign_p2f_<dataset>_a100_<instrument>_fp64``) and the
  references above sma (``submit_campaign_p2f_references_ral_cpu_fp64``), refusing to run until
  batch 0's walls are in (:data:`WALL_BASIS_STATUS` is ``provisional``).

Stdlib only; nothing here imports the modelling stack except :func:`prepare_dataset`, which the
submits call to simulate a missing dataset and cache its adapt image (and, for the datacube, the
NumPy W~ preload) under a lock.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

from _script_routes import load_routes  # noqa: E402

#: The release every cell is measured on (2b's pin, the human 2026-10-28), and the private RAL
#: checkout of the five libraries at that tag (``PYAUTO_PRIVATE_BASE``).
PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
#: The shared virtualenv (third-party packages only; finufft 2.5.1 since 2026-10-26).
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#475"

DATASET_TYPES = ("multi_dataset", "datacube")
VARIANT = "delaunay"
#: Planned instruments (the registry): the multi-dataset cell defaults to hst; the datacube runs
#: on the three interferometer presets it was planned on, ordered by visibilities (sma 190, alma
#: 1e6, alma_high 5e6).
INSTRUMENTS = {"multi_dataset": ("hst",), "datacube": ("sma", "alma", "alma_high")}
#: The section 3.3 setups per host class (``catalogue/provisional_designations.json``):
#: multi_dataset reads the imaging pixelized table, datacube the interferometer pixelized one.
SETUPS = {
    "multi_dataset": {
        "hpc_ral_cpu": ("numba_cpu_sparse_fp64", "jax_cpu_sparse_fp64"),
        "hpc_a100": ("jax_gpu_dense_fp64", "jax_gpu_sparse_fp64", "jax_cpu_sparse_fp64"),
    },
    "datacube": {
        "hpc_ral_cpu": (
            "numba_cpu_sparse_fp64_wtilde",
            "jax_cpu_sparse_fp64_wtilde",
            "jax_cpu_dense_fp64_nufft",
        ),
        "hpc_a100": ("jax_gpu_sparse_fp64_wtilde", "jax_gpu_dense_fp64_nufft"),
    },
}
#: Decision D1: the dense datacube NUFFT route is planned at sma only, as 2b's R1(a) for the
#: interferometer (each channel's dense transform is the interferometer's: 20.5 GB at alma,
#: 102 GB at alma_high). ``catalogue/provisional_designations.json`` scopes it the same way.
DENSE_ONLY_AT = ("sma",)
MEASUREMENTS = ("runtime", "breakdown")
#: ``--config-name`` per (host class, device): the host tier a row's witness judges against.
CONFIG_NAMES = {
    ("hpc_ral_cpu", "cpu"): "hpc_ral_cpu_fp64",
    ("hpc_a100", "gpu"): "hpc_a100_fp64",
    ("hpc_a100", "cpu"): "hpc_a100_cpu_fp64",
}
#: ``--n-instances`` of the datacube numba legs: 2b's R4 per instrument.
NUMBA_N_INSTANCES = {"sma": 20, "alma": 20, "alma_high": 10}
#: Decision D2: channels per cube (``datacube_standard.N_CHANNELS``); decision D6: datasets per
#: multi-dataset fit (``multi_dataset_standard.N_DATASETS``). The scripts default to these; the
#: submits pass them explicitly so a row's size is visible in the job.
N_CHANNELS = 34
N_DATASETS = 2

#: Decision D9: batch 0 runs the legs that bound every other one -- alma_high for the datacube's
#: W~ and numba setups (the most visibilities), sma for its dense setups (sma only, D1), and every
#: multi_dataset leg (hst is its only instrument; 2a measured no hst Delaunay leg).
BOUNDING_INSTRUMENT = {
    ("datacube", "sparse"): "alma_high",
    ("datacube", "dense"): "sma",
    ("multi_dataset", "sparse"): "hst",
    ("multi_dataset", "dense"): "hst",
}
#: Batch 0's declared per-leg wall guesses (``source: unmeasured``, 3x floor) and ``--time``.
BATCH0_WALL_GUESS_S = {"ral_cpu": 14000, "a100": 7000, "reference": 14000}
BATCH0_TIME = {"ral_cpu": "12:00:00", "a100": "6:00:00", "reference": "12:00:00"}
#: Batch 0's measured legs, once it has run: cell -> (walls.tsv WALL, sacct Elapsed, task).
BATCH0_WALLS: dict = {}
BATCH0_LIMITATIONS: dict = {}
#: ``provisional`` until :data:`BATCH0_WALLS` holds batch 0; batch 1 then renders unmeasured rows,
#: :data:`BATCH1_PROVISIONAL_TIME` and a submit that refuses to run
#: (``P2F_ALLOW_PROVISIONAL_WALL=1`` overrides).
WALL_BASIS_STATUS = "provisional"
BATCH1_PROVISIONAL_TIME = {"ral_cpu": "12:00:00", "a100": "2-00:00:00", "reference": "12:00:00"}
WALL_HEADROOM = 1.25
WALL_ROUND_S = 1800
WALL_MIN_SLACK_S = 900

#: Declared ``--mem`` (GB) per instrument, estimates from the dataset sizes (not measured): the
#: cube holds 34 copies of the visibilities (alma_high 5e6 x 16 B = 80 MB each, so ~2.7 GB) beside
#: 2b's single-channel peaks (sma 30.8 GiB, jvla 21.4 GiB in 2b batch 0); hst imaging x2 is
#: below 2a's hst peak x2 (10.2 GiB). Batch 0's sacct MaxRSS replaces them before batch 1.
RAL_MEM_GB = {"hst": 32, "sma": 64, "alma": 64, "alma_high": 128}
A100_MEM_GB = {"hst": 64, "sma": 64, "alma": 64, "alma_high": 64}
#: 2b's chunked dense reference, alma_high (5e5-visibility chunks, peak ~31 GiB at jvla).
REFERENCE_MEM_GB = 64
MEM_STEPS_GB = (16, 32, 64, 128, 256)
MEM_HEADROOM = 1.5
BATCH1_MEM_FLOOR_GB = 32
#: Batch 0's peak ``.batch`` MaxRSS (KiB) per instrument, once it has run.
BATCH0_PEAK_RSS_KIB: dict = {}

_ROUTES = {row["path"]: row["legacy"] for row in load_routes(ROOT)["routes"]}


@dataclass(frozen=True, order=True)
class Task:
    """One leg. ``kind``: ``cell`` (a planned coverage cell,
    ``<dataset>/delaunay/<instrument> <setup_id>@<host_class> [<measurement>]``) or ``reference``
    (2b's chunked dense reference of delaunay x instrument, finalising the datacube rows)."""

    dataset: str
    instrument: str
    setup_id: str
    host_class: str
    measurement: str
    kind: str = "cell"
    batch: int = 1

    variant = VARIANT

    @property
    def backend(self) -> str:
        return self.setup_id.split("_")[0]

    @property
    def device(self) -> str:
        return self.setup_id.split("_")[1]

    @property
    def operator(self) -> str:
        return self.setup_id.split("_")[2]

    @property
    def cell(self) -> str:
        where = f"{self.dataset}/{VARIANT}/{self.instrument}"
        if self.kind == "reference":
            return f"{where} dense_reference@{self.host_class} [parity]"
        return f"{where} {self.setup_id}@{self.host_class} [{self.measurement}]"

    @property
    def script(self) -> str:
        """The leg's script, relative to ``scripts/`` and without ``.py``."""
        if self.kind == "reference":
            return f"datacube/{VARIANT}/parity_reference"
        suffix = "_numba" if self.backend == "numba" else ""
        return f"{self.dataset}/{VARIANT}/likelihood_{self.measurement}{suffix}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id ``wall/check_submits.py`` reads off the script's legacy route:
        ``<legacy dataset folder>/<legacy stem>`` (the datacube JAX scripts' legacy routes sit
        under ``scripts/interferometer/``, so theirs read ``interferometer/delaunay``)."""
        legacy = _ROUTES[f"scripts/{self.script}.py"]
        return f"{legacy.split('/')[1]}/{Path(legacy).stem}"

    @property
    def config_name(self) -> str:
        return CONFIG_NAMES[(self.host_class, self.device)]

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (``run_leg`` adds those)."""
        if self.kind == "reference":
            # Relative: FLAGS reach the command line unquoted, so "$RESULTS_ROOT" never expands.
            return f"--results-root output/p2f_batch{self.batch} --config-name {self.config_name}"
        flags = ["--standard-size"]
        if self.dataset == "datacube":
            flags += ["--n-channels", str(N_CHANNELS)]
            if self.backend == "jax" and self.operator == "sparse" and self.measurement == "runtime":
                flags.append("--sparse")
            if self.backend == "jax" and self.operator == "dense" and self.measurement == "breakdown":
                flags.append("--dense")
            if self.backend == "numba":
                flags += ["--n-instances", str(NUMBA_N_INSTANCES[self.instrument])]
        else:
            flags += ["--n-datasets", str(N_DATASETS)]
            if self.backend == "jax" and self.operator == "sparse":
                flags.append("--sparse")
        flags += ["--config-name", self.config_name]
        return " ".join(flags)

    @property
    def output(self) -> str:
        """The leg's output, relative to the results root (a reference's sits beside it)."""
        if self.kind == "reference":
            return f"../p2b_references/interferometer_{VARIANT}_{self.instrument}.json"
        return (
            f"results/{self.dataset}/{VARIANT}/{self.instrument}/{self.setup_id}/"
            f"{self.host_class}/{self.measurement}.json"
        )

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS`` for the leg (numba / NumPy and jax_cpu legs never touch a GPU)."""
        return "cuda,cpu" if self.device == "gpu" else "cpu"

    @property
    def bound(self) -> str:
        """``dense`` (datacube: sma only) or ``sparse`` (W~, numba, references)."""
        return "dense" if self.kind == "cell" and self.operator == "dense" else "sparse"


def planned_tasks() -> list[Task]:
    """Every planned cell of the campaign, sorted (32 under D1: 18 RAL CPU + 14 A100)."""
    tasks = []
    for dataset in DATASET_TYPES:
        for instrument in INSTRUMENTS[dataset]:
            for host_class, setups in SETUPS[dataset].items():
                for setup_id in setups:
                    if (
                        dataset == "datacube"
                        and "_dense_" in setup_id
                        and instrument not in DENSE_ONLY_AT
                    ):
                        continue
                    for measurement in MEASUREMENTS:
                        tasks.append(Task(dataset, instrument, setup_id, host_class, measurement))
    return sorted(tasks)


def reference_tasks(instruments=("alma", "alma_high"), batch: int = 1) -> list[Task]:
    """The datacube references above sma (D3): 2b's chunked dense reference task per instrument,
    reused when 2b's cache is valid, then the datacube rows' ``parity.json`` finalised."""
    return sorted(
        Task("datacube", i, "numba_cpu_dense_fp64", "hpc_ral_cpu", "parity", "reference", batch)
        for i in instruments
    )


def batch1_ral_cpu(dataset: str, instrument: str) -> list[Task]:
    return [
        t
        for t in planned_tasks()
        if t.host_class == "hpc_ral_cpu" and (t.dataset, t.instrument) == (dataset, instrument)
    ]


def batch1_a100(dataset: str, instrument: str) -> list[Task]:
    return [
        t
        for t in planned_tasks()
        if t.host_class == "hpc_a100" and (t.dataset, t.instrument) == (dataset, instrument)
    ]


def _is_bounding(task: Task) -> bool:
    return task.instrument == BOUNDING_INSTRUMENT[(task.dataset, task.bound)]


def batch0_ral_cpu() -> list[Task]:
    return [
        replace(t, batch=0)
        for t in planned_tasks()
        if t.host_class == "hpc_ral_cpu" and _is_bounding(t)
    ]


def batch0_a100() -> list[Task]:
    return [
        replace(t, batch=0) for t in planned_tasks() if t.host_class == "hpc_a100" and _is_bounding(t)
    ]


def batch0_references() -> list[Task]:
    return reference_tasks(("alma_high",), batch=0)


def basis_leg(task: Task) -> Task:
    """The batch-0 leg bounding ``task``: same script, setup, host class and measurement on the
    bounding instrument (alma_high for the datacube W~, numba and reference legs; sma for its
    dense legs; hst for multi_dataset)."""
    return replace(task, instrument=BOUNDING_INSTRUMENT[(task.dataset, task.bound)], batch=0)


def measured_wall(task: Task) -> tuple[int, str]:
    """(seconds, batch-0 task) for ``task``: the larger of its basis leg's WALL and Elapsed."""
    wall, elapsed, ref = BATCH0_WALLS[basis_leg(task).cell]
    return max(wall, elapsed), ref


WALL_BASIS = _common.WallBasis(
    measured_wall=measured_wall,
    basis_leg=basis_leg,
    limitations=BATCH0_LIMITATIONS,
    walls=BATCH0_WALLS,
    headroom=WALL_HEADROOM,
)


def slurm_time(seconds: float) -> str:
    return _common.slurm_time(
        seconds, headroom=WALL_HEADROOM, slack_s=WALL_MIN_SLACK_S, round_s=WALL_ROUND_S
    )


def mem_gb_measured(instruments, minimum: int = BATCH1_MEM_FLOOR_GB) -> int:
    """Batch 1's ``--mem`` once batch 0 has run: the next step at or above MEM_HEADROOM x the
    bounding instrument's peak (alma and sma datacube legs take alma_high's)."""

    def peak(instrument):
        if instrument in BATCH0_PEAK_RSS_KIB:
            return BATCH0_PEAK_RSS_KIB[instrument][0] / 2**20
        return max(kib for kib, _ in BATCH0_PEAK_RSS_KIB.values()) / 2**20

    top = max(peak(i) for i in instruments)
    return next(step for step in MEM_STEPS_GB if step >= max(minimum, MEM_HEADROOM * top))


# --------------------------------------------------------------------------- dataset preflight


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """Simulate the leg's dataset if missing and cache its inputs, under a lock.

    hst (multi_dataset): 2a's imaging preflight (``imaging_p2a.prepare_dataset``). The datacube
    instruments: 2b's interferometer preflight (``interferometer_p2b.prepare_dataset``: the
    dataset, ``lensed_source.fits`` and the NumPy W~ preload), since every channel of a cube is
    the instrument's interferometer dataset. Both lock per instrument, so array tasks never race.
    """
    if instrument in INSTRUMENTS["multi_dataset"]:
        import imaging_p2a

        imaging_p2a.prepare_dataset(instrument, root)
    else:
        import interferometer_p2b

        interferometer_p2b.prepare_dataset(instrument, root)


# --------------------------------------------------------------------------- submit rendering


def _campaign() -> _common.Campaign:
    return _common.Campaign(
        pin_var="P2F_PIN",
        pin=PIN,
        hpc_base=HPC_BASE,
        private_base=PRIVATE_BASE,
        logs="p2f_logs",
        prefix="p2f",
        module="multi_dataset_datacube_p2f",
        allow_var="P2F_ALLOW_PROVISIONAL_WALL",
    )


def _instrument_order(instrument: str) -> int:
    order = INSTRUMENTS["multi_dataset"] + INSTRUMENTS["datacube"]
    return order.index(instrument)


def _mem(tasks: list[Task], device: str, batch: int) -> tuple[int, str]:
    """(``--mem`` GB, the header lines saying where it came from)."""
    instruments = sorted({t.instrument for t in tasks}, key=_instrument_order)
    if tasks[0].kind == "reference":
        return REFERENCE_MEM_GB, (
            f"# MEMORY. --mem={REFERENCE_MEM_GB}gb, 2b's reference request: S^2 + one 5e5-visibility"
            "\n# chunk of T (2b batch 0 peaked at 30.5 GiB on jvla)."
        )
    if batch == 1 and WALL_BASIS_STATUS == "measured":
        mem = mem_gb_measured(instruments)
        return mem, (
            f"# MEMORY. --mem={mem}gb from batch 0's sacct MaxRSS (bounding instrument) x "
            f"{MEM_HEADROOM},\n# at least {BATCH1_MEM_FLOOR_GB} GB."
        )
    table = RAL_MEM_GB if device == "ral_cpu" else A100_MEM_GB
    mem = max(table[i] for i in instruments)
    return mem, (
        f"# MEMORY. --mem={mem}gb, a declared estimate for {', '.join(instruments)} (the largest"
        "\n# instrument decides; multi_dataset_datacube_p2f.RAL_MEM_GB / A100_MEM_GB). Batch 0's"
        "\n# sacct MaxRSS replaces it before batch 1."
    )


def render(name: str, tasks: list[Task], *, kind: str, batch: int, title: str) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) / ``a100_job`` / ``a100_array``."""
    reference = tasks[0].kind == "reference"
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    budget = "reference" if reference else device
    sequential = kind == "a100_job"
    measured = batch == 1 and WALL_BASIS_STATUS == "measured"
    if batch == 0:
        guess, time = BATCH0_WALL_GUESS_S[budget], BATCH0_TIME[budget]
    elif measured:
        guess = None
        time = slurm_time(_common.batch_wall_s(tasks, WALL_BASIS, sequential=sequential))
    else:
        guess, time = None, BATCH1_PROVISIONAL_TIME[budget]
    results_root = f"$AP_ROOT/output/p2f_batch{batch}"
    mem, mem_line = _mem(tasks, device, batch)
    n = len(tasks)
    datasets = sorted({t.dataset for t in tasks})
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        f"# Likelihood-lifecycle phase 2f multi_dataset + datacube campaign ({ISSUE}), batch "
        f"{batch}: {n} leg(s).",
        "# DRAFT FOR REVIEW: the decisions D1-D9 (wiki/campaigns/datacube/likelihood_lifecycle_p2f.md)",
        "# await the human; do not submit before the human's batch go.",
        "# GENERATED by scripts/misc/campaigns/multi_dataset_datacube_p2f.py --write; edit the",
        "# generator, not this file (the campaign test fails on a hand edit). The batch procedure is",
        "# in scripts/misc/campaigns/README.md.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at tag {PIN} ($PYAUTO_PRIVATE_BASE,",
        f"# default {PRIVATE_BASE}) is prepended to PYTHONPATH over the shared venv",
        f"# ($PYAUTO_HPC_BASE, default {HPC_BASE}, third-party packages only; finufft 2.5.1",
        "# serves the NumPy NUFFT). The job refuses to run if a library imports from anywhere",
        "# else or sits at another tag, and after each leg it checks the row's library_revisions",
        "# equal the checkout's HEADs. The shared mirror is never pulled or used. AP_ROOT is the",
        "# checkout this was submitted from (hpc/<batch>/../..).",
        "#",
    ]
    if reference:
        header += [
            "# REFERENCES (D3). Each task computes (or reuses, when 2b's cache in",
            "# output/p2b_references/ is valid) the visibility-chunked eager NumPy dense log evidence",
            "# of the delaunay standard cell at one instrument, then rewrites parity.json of every",
            f"# datacube row of that instrument under output/p2f_batch{batch}/ that carries a cube",
            "# parity_candidate (the cube reference follows from the channel reference's terms).",
        ]
    else:
        header += [
            "# CELLS. Every leg runs one standard-size cell (--standard-size) and writes its row,",
            "# fit.png and parity.json into the mirrored cell named by --output-path under the",
            f"# gitignored output/p2f_batch{batch}/ scratch tree, never in place over committed rows.",
        ]
        if "datacube" in datasets:
            header += [
                f"# datacube (D2): 2b's delaunay interferometer standard cell in each of {N_CHANNELS}",
                "# channels (shared uv, noise map, mask and lens model; channel visibilities scaled",
                "# by an emission line, a_0 = 1), FactorGraphModel of AnalysisInterferometer",
                "# (shared_preloads=True). The datacube JAX scripts' legacy routes sit under",
                "# scripts/interferometer/, so their WALL-BASIS rows read interferometer/delaunay.",
            ]
        if "multi_dataset" in datasets:
            header += [
                f"# multi_dataset (D6): 2a's hst delaunay imaging standard cell in each of {N_DATASETS}",
                "# datasets (data and noise map x 1, x 0.5), FactorGraphModel of AnalysisImaging.",
            ]
    header += ["#", mem_line]
    if kind == "ral_cpu":
        header += [
            "# One leg per array task on `ral` (CPU work never goes on `gpu`). numba legs pin their",
            "# BLAS and numba threads to 1; JAX CPU legs get NPROC=8.",
        ]
    elif kind == "a100_job":
        header += [
            "# One A100 job runs these cells in turn. A jax_cpu_sparse leg runs on the GPU node's",
            "# CPU inside this GPU job (JAX_PLATFORMS=cpu, NPROC=8), never a CPU array on `gpu`.",
        ]
    else:
        header += [
            "# One leg per array task, each with --gres=gpu:1, throttled to %2 so the timing arm",
            "# holds at most two A100s. A jax_cpu_sparse leg runs on the GPU node's CPU inside its",
            "# GPU task (JAX_PLATFORMS=cpu, NPROC=8).",
        ]
    header += ["#"]
    if batch == 0:
        header += [
            "# BATCH 0 = TIMING ARM (D9). It runs the legs that bound every other one: alma_high for",
            "# the datacube W~ and numba legs, sma for its dense legs (sma only, D1), and every",
            "# multi_dataset leg (hst only). Each leg's wall lands in output/p2f_logs/walls.tsv;",
            "# batch 1's WALL-BASIS rows cite them and its --time is set from them.",
            "#",
            f"# WALL-BASIS: nothing measured at the standard size yet; {guess} s per leg is a declared",
            "# guess (3x floor). The timing arm exists to find out.",
        ]
    elif measured:
        header += [
            "# WALL-BASIS: measured by batch 0. Each leg's wall is that of its batch-0 leg of the",
            "# same script, setup and host class on the bounding instrument, the larger of the",
            f"# walls.tsv WALL and sacct Elapsed; --time is {WALL_HEADROOM}x the total plus",
            f"# {WALL_MIN_SLACK_S // 60} min of slack, rounded up to the next {WALL_ROUND_S // 60} min.",
        ]
    else:
        header += [
            "# WALL-BASIS: PROVISIONAL. No leg has a standard-size wall yet; batch 0 measures them.",
            "# Every row is `unmeasured` until batch 0 lands, and the job refuses to run while",
            "# WALL_BASIS_STATUS=provisional (export P2F_ALLOW_PROVISIONAL_WALL=1 overrides it).",
        ]
    header += _common.wall_rows(
        tasks, device, guess, measured=measured, sequential=sequential, basis=WALL_BASIS
    )
    job = "p2f_" + name.split("submit_campaign_p2f_", 1)[1].removesuffix("_fp64")
    sbatch = _common.sbatch_block(
        job,
        kind,
        n,
        mem=mem,
        time=time,
        throttle=("%8" if batch == 0 else "%16") if kind == "ral_cpu" else None,
    )
    campaign = _campaign()
    return _common.assemble(
        header,
        sbatch,
        _common.provisional_guard(WALL_BASIS_STATUS, campaign.allow_var) if batch == 1 else None,
        _common.preamble(campaign, name),
        _common.manifest_arrays(tasks, results_root, module="multi_dataset_datacube_p2f"),
        _common.run_block(kind),
    )


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text}."""
    out = {}

    def add(folder, name, tasks, **kwargs):
        out[f"hpc/{folder}/{name}"] = render(name, tasks, **kwargs)

    add(
        "batch_cpu",
        "submit_campaign_p2f_batch0_ral_cpu_fp64",
        batch0_ral_cpu(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm, RAL CPU: the bounding hpc_ral_cpu multi_dataset + datacube cells.",
    )
    add(
        "batch_gpu",
        "submit_campaign_p2f_batch0_a100_fp64",
        batch0_a100(),
        kind="a100_array",
        batch=0,
        title="Batch-0 timing arm, A100: the bounding hpc_a100 multi_dataset + datacube cells.",
    )
    add(
        "batch_cpu",
        "submit_campaign_p2f_batch0_references_ral_cpu_fp64",
        batch0_references(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm (optional), RAL CPU: the alma_high delaunay parity reference.",
    )
    for dataset in DATASET_TYPES:
        for instrument in INSTRUMENTS[dataset]:
            add(
                "batch_cpu",
                f"submit_campaign_p2f_{dataset}_ral_cpu_{instrument}_fp64",
                batch1_ral_cpu(dataset, instrument),
                kind="ral_cpu",
                batch=1,
                title=f"RAL CPU array: the hpc_ral_cpu {dataset} cells of {instrument}.",
            )
            add(
                "batch_gpu",
                f"submit_campaign_p2f_{dataset}_a100_{instrument}_fp64",
                batch1_a100(dataset, instrument),
                kind="a100_job",
                batch=1,
                title=f"A100 job: the hpc_a100 {dataset} cells of {instrument}.",
            )
    add(
        "batch_cpu",
        "submit_campaign_p2f_references_ral_cpu_fp64",
        reference_tasks(),
        kind="ral_cpu",
        batch=1,
        title="RAL CPU array: the delaunay parity references above sma (alma, alma_high).",
    )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="(re)render the submits")
    mode.add_argument("--check", action="store_true", help="exit 1 if a submit is stale")
    mode.add_argument("--list", action="store_true", help="print the manifest")
    args = parser.parse_args(argv)
    if args.list:
        for task in planned_tasks() + reference_tasks():
            print(f"{task.cell:<80} {task.script}.py {task.flags}")
        return 0
    return _common.write_or_check(
        submits(), ROOT, args=args, generator="multi_dataset_datacube_p2f.py"
    )


if __name__ == "__main__":
    sys.exit(main())
