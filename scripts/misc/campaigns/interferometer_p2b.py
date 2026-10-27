"""Likelihood-lifecycle phase 2b: the interferometer campaign's task manifest and its SLURM submits.

autolens_profiling#469 measures the 60 planned interferometer cells (``interferometer/{mge,
rectangular, delaunay}``; ``dashboard/coverage.json``) at the standard size on the pinned release
2026.10.25.1 (ruling R5, changed by the human 2026-10-26), from a private RAL checkout. This module
is the one copy of that work list: every cell becomes one :class:`Task` (script, flags, output
path), and the submits under ``hpc/`` are rendered from the tasks, with the shared machinery in
``_common.py`` (the same as phase 2a's ``imaging_p2a.py``)::

    python scripts/misc/campaigns/interferometer_p2b.py --write   # (re)render the submits
    python scripts/misc/campaigns/interferometer_p2b.py --check   # exit 1 if a submit is stale
    python scripts/misc/campaigns/interferometer_p2b.py --list    # print the manifest

The submits (README.md beside this file has the batch procedure):

- batch 0, the timing arm (ruling R8): ``hpc/batch_cpu/submit_campaign_p2b_batch0_ral_cpu_fp64``
  (``--partition=ral`` array of 14), ``hpc/batch_gpu/submit_campaign_p2b_batch0_a100_fp64``
  (``--gres=gpu:1`` array of 10, ``%2``) and the optional jvla dense-reference tasks
  ``hpc/batch_cpu/submit_campaign_p2b_batch0_references_ral_cpu_fp64`` (2), writing into the
  gitignored ``output/p2b_batch0/`` and ``output/p2b_references/``;
- batch 1, one ``--partition=ral`` array per instrument
  (``submit_campaign_p2b_interferometer_ral_cpu_<instrument>_fp64``), one ``--gres=gpu:1`` job per
  instrument (``submit_campaign_p2b_interferometer_a100_<instrument>_fp64``) and the dense
  references above sma (``submit_campaign_p2b_references_ral_cpu_fp64``), refusing to run until
  batch 0's walls are in (:data:`WALL_BASIS_STATUS`);
- the matrix-free experiment (``hpc/batch_cpu/submit_campaign_p2b_matrixfree_ral_cpu_fp64``,
  :func:`matrixfree_tasks`), on its own private checkout of unreleased library main at declared
  SHAs, writing ``experiments/`` and ``scans/`` rows only (never current).

Stdlib only; nothing here imports the modelling stack except :func:`prepare_dataset`, which the
submits call to simulate a missing dataset, cache its adapt image and its NumPy W~ preload under
a lock.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

from _script_routes import load_routes  # noqa: E402

#: The release every cell is measured on (ruling R5, changed by the human 2026-10-26), and where
#: the private RAL checkout of the five libraries at that tag lives (``PYAUTO_PRIVATE_BASE``).
PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
#: The shared virtualenv (third-party packages only; finufft 2.5.1 installed 2026-10-26).
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#469"

#: Ordered by visibility count (sma 190, alma 1e6, alma_high 5e6, jvla 2.5e7).
INSTRUMENTS = ("sma", "alma", "alma_high", "jvla")
#: Planned instruments per variant: the registry (ruling R9; mge at sma only).
VARIANT_INSTRUMENTS = {
    "mge": ("sma",),
    "rectangular": INSTRUMENTS,
    "delaunay": INSTRUMENTS,
}
PIXELIZED = ("rectangular", "delaunay")
#: The §3.3 setups per host class (``catalogue/provisional_designations.json``). No DFT cell
#: (ruling R2); dense pixelized NUFFT is planned at sma only (ruling R1(a), :data:`DENSE_ONLY_AT`).
SETUPS = {
    "mge": {
        "hpc_ral_cpu": ("jax_cpu_dense_fp64_nufft",),
        "hpc_a100": ("jax_gpu_dense_fp64_nufft",),
    },
    "pixelized": {
        "hpc_ral_cpu": (
            "numba_cpu_sparse_fp64_wtilde",
            "jax_cpu_sparse_fp64_wtilde",
            "jax_cpu_dense_fp64_nufft",
        ),
        "hpc_a100": ("jax_gpu_sparse_fp64_wtilde", "jax_gpu_dense_fp64_nufft"),
    },
}
DENSE_ONLY_AT = ("sma",)
MEASUREMENTS = ("runtime", "breakdown")
CONFIG_NAMES = {"hpc_ral_cpu": "hpc_ral_cpu_fp64", "hpc_a100": "hpc_a100_fp64"}
#: ``--n-instances`` of the numba legs (ruling R4): the jvla direct_conv arm is ~540 s a call.
NUMBA_N_INSTANCES = {"sma": 20, "alma": 20, "alma_high": 10, "jvla": 5}

#: The instrument whose batch-0 leg bounds a setup on every instrument: the sparse and numba
#: legs scale with the mask extent and visibilities, so jvla bounds them (and was never measured
#: on CPU); the dense and MGE legs exist only at sma.
BOUNDING_INSTRUMENT = {"sparse": "jvla", "dense": "sma"}
#: Batch 0's declared per-leg wall guesses (``source: unmeasured``, 3x floor) and ``--time``.
BATCH0_WALL_GUESS_S = {"ral_cpu": 14000, "a100": 7000, "reference": 28000}
BATCH0_TIME = {"ral_cpu": "12:00:00", "a100": "6:00:00", "reference": "24:00:00"}

#: Batch 0's measured legs (filled by the walls PR): cell -> (walls.tsv WALL, sacct Elapsed,
#: array task), keyed by the batch-0 leg's :attr:`Task.cell`.
BATCH0_WALLS: dict = {}
#: Batch-0 legs that cannot run, recorded as a known limitation by human ruling.
BATCH0_LIMITATIONS: dict = {}
#: ``measured`` once :data:`BATCH0_WALLS` holds batch 0 and the batch-1 rows cite it; while
#: ``provisional`` batch 1 renders unmeasured rows, :data:`BATCH1_PROVISIONAL_TIME` and a submit
#: that refuses to run (``P2B_ALLOW_PROVISIONAL_WALL=1`` overrides).
WALL_BASIS_STATUS = "provisional"
BATCH1_PROVISIONAL_TIME = {"ral_cpu": "12:00:00", "a100": "2-00:00:00", "reference": "24:00:00"}
WALL_HEADROOM = 1.25
WALL_ROUND_S = 1800
WALL_MIN_SLACK_S = 900

#: Declared ``--mem`` (GB) per instrument, an estimate from the dataset sizes (survey A8): the
#: jvla mapping matrix alone is 384,852 x 1,280 x 8 B = 3.9 GB, the dataset ~1.2 GB, the JAX CPU
#: W~ FFT batches and compile temporaries unmeasured. Batch 0's MaxRSS replaces them.
RAL_MEM_GB = {"sma": 32, "alma": 32, "alma_high": 64, "jvla": 128}
A100_MEM_GB = {"sma": 32, "alma": 64, "alma_high": 64, "jvla": 64}
#: The chunked dense reference (5e5-visibility chunks, peak ~25 GB at S = 1280).
REFERENCE_MEM_GB = 64
MEM_STEPS_GB = (16, 32, 64, 128, 256)
#: Batch 0's peak MaxRSS (KiB) per instrument of the cell legs, and of the reference tasks, with
#: the task that reached it (filled by the walls PR).
BATCH0_PEAK_RSS_KIB: dict = {}
REFERENCE_PEAK_RSS_KIB: dict = {}
MEM_HEADROOM = 1.5
BATCH1_MEM_FLOOR_GB = 32

#: The matrix-free experiment (human 2026-10-26): PyAutoArray#658 merged after 2026.10.25.1, so it
#: runs from a private checkout of library main at these SHAs (origin/main, 2026-10-27), labelled
#: unreleased. PyAutoArray 7eba0c07 is also tag 2026.10.26.1 on PyPI (open question for the human).
MF_PRIVATE_BASE = "/mnt/ral/jnightin/profiling/PyAuto_main_7eba0c07"
MF_LIBRARY_SHAS = {
    "PyAutoNerves": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
    "PyAutoFit": "8740092dc12b809847be07255ff427b50eae059e",
    "PyAutoArray": "7eba0c07872a6c83ac2d12a72c4b6613cb3b6a0f",
    "PyAutoGalaxy": "a1080929d52e5f2546adf8583b4b21040b201c13",
    "PyAutoLens": "a8eb094e539185ea2485beda25e7411df176da65",
}
MF_STANDARD_SIDE = 35
MF_SCAN_INSTRUMENT = "alma"
MF_SCAN_SIDES = (64, 128, 192)
MF_SETUP = ("numba_cpu_sparse_fp64_wtilde", "hpc_ral_cpu")
MF_EXPERIMENT = "numpy_cpu_matrixfree"
MF_WALL_GUESS_S = 14000
MF_TIME = "12:00:00"
MF_MEM_GB = 128

_ROUTES = {row["path"]: row["legacy"] for row in load_routes(ROOT)["routes"]}


@dataclass(frozen=True, order=True)
class Task:
    """One leg. ``kind``: ``cell`` (a planned coverage cell,
    ``interferometer/<variant>/<instrument> <setup_id>@<host_class> [<measurement>]``),
    ``reference`` (the chunked dense parity reference of one variant x instrument) or
    ``matrixfree`` (the experiment at ``mesh_side`` x ``mesh_side``)."""

    variant: str
    instrument: str
    setup_id: str
    host_class: str
    measurement: str
    kind: str = "cell"
    mesh_side: int | None = None
    batch: int = 1

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
        where = f"interferometer/{self.variant}/{self.instrument}"
        if self.kind == "reference":
            return f"{where} dense_reference@{self.host_class} [parity]"
        if self.kind == "matrixfree":
            side = self.mesh_side
            return f"{where} {MF_EXPERIMENT}_{side}x{side}@{self.host_class} [runtime]"
        return f"{where} {self.setup_id}@{self.host_class} [{self.measurement}]"

    @property
    def script(self) -> str:
        """The leg's script, relative to ``scripts/`` and without ``.py``."""
        if self.kind == "reference":
            return "interferometer/pixelized/parity_reference"
        if self.kind == "matrixfree":
            return "interferometer/rectangular/likelihood_runtime_matrixfree"
        suffix = "_numba" if self.backend == "numba" else ""
        return f"interferometer/{self.variant}/likelihood_{self.measurement}{suffix}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id: ``interferometer/<legacy script stem>``."""
        legacy = _ROUTES[f"scripts/{self.script}.py"]
        return f"interferometer/{Path(legacy).stem}"

    @property
    def config_name(self) -> str:
        return CONFIG_NAMES[self.host_class]

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (``run_leg`` adds those)."""
        if self.kind == "reference":
            # A relative --results-root (the job runs from AP_ROOT): FLAGS reach the command
            # line unquoted, so a "$RESULTS_ROOT" here would never expand.
            return (
                f"--likelihood {self.variant} --results-root output/p2b_batch{self.batch} "
                f"--config-name {self.config_name}"
            )
        if self.kind == "matrixfree":
            return (
                f"--mesh-side {self.mesh_side} --n-instances {NUMBA_N_INSTANCES[self.instrument]} "
                f"--config-name {self.config_name}"
            )
        flags = []
        if self.variant in PIXELIZED:
            flags.append("--standard-size")
        if self.backend == "jax" and self.operator == "sparse" and self.measurement == "runtime":
            flags.append("--sparse")
        if self.backend == "jax" and self.operator == "dense" and self.measurement == "breakdown":
            if self.variant in PIXELIZED:
                flags.append("--dense")
        if self.backend == "numba":
            flags += ["--n-instances", str(NUMBA_N_INSTANCES[self.instrument])]
        flags += ["--config-name", self.config_name]
        return " ".join(flags)

    @property
    def output(self) -> str:
        """The leg's output, relative to the results root (a reference's sits beside it)."""
        if self.kind == "reference":
            return f"../p2b_references/interferometer_{self.variant}_{self.instrument}.json"
        cell = (
            f"results/interferometer/{self.variant}/{self.instrument}/{self.setup_id}/"
            f"{self.host_class}"
        )
        if self.kind == "matrixfree":
            if self.mesh_side == MF_STANDARD_SIDE:
                return f"{cell}/experiments/{MF_EXPERIMENT}/runtime.json"
            side = self.mesh_side
            return f"{cell}/scans/{MF_EXPERIMENT}_{side}x{side}/runtime.json"
        return f"{cell}/{self.measurement}.json"

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS`` for the leg (numba / NumPy legs never touch a JAX device)."""
        return "cuda,cpu" if self.device == "gpu" else "cpu"

    @property
    def bound(self) -> str:
        """``sparse`` (bounded at jvla) or ``dense`` (sma only)."""
        if self.kind == "reference":
            return "sparse"
        if self.variant == "mge" or self.operator == "dense":
            return "dense"
        return "sparse"


def planned_tasks() -> list[Task]:
    """Every planned interferometer cell of the campaign, sorted (60: 38 RAL CPU + 22 A100)."""
    tasks = []
    for variant, instruments in VARIANT_INSTRUMENTS.items():
        table = SETUPS["mge" if variant == "mge" else "pixelized"]
        for instrument in instruments:
            for host_class, setups in table.items():
                for setup_id in setups:
                    if (
                        variant in PIXELIZED
                        and "_dense_" in setup_id
                        and instrument not in DENSE_ONLY_AT
                    ):
                        continue
                    for measurement in MEASUREMENTS:
                        tasks.append(Task(variant, instrument, setup_id, host_class, measurement))
    return sorted(tasks)


def reference_tasks(instruments=INSTRUMENTS[1:], batch: int = 1) -> list[Task]:
    """The chunked dense-reference tasks: one per (variant, instrument above sma), ruling R6."""
    return sorted(
        Task(v, i, "numba_cpu_dense_fp64", "hpc_ral_cpu", "parity", "reference", batch=batch)
        for v in PIXELIZED
        for i in instruments
    )


def matrixfree_tasks() -> list[Task]:
    """The matrix-free experiment: 35x35 on every instrument plus the alma mesh scan."""
    sid, host = MF_SETUP
    tasks = [
        Task("rectangular", i, sid, host, "runtime", "matrixfree", MF_STANDARD_SIDE)
        for i in INSTRUMENTS
    ]
    tasks += [
        Task("rectangular", MF_SCAN_INSTRUMENT, sid, host, "runtime", "matrixfree", side)
        for side in MF_SCAN_SIDES
    ]
    return tasks


def batch1_ral_cpu(instrument: str | None = None) -> list[Task]:
    return [
        t
        for t in planned_tasks()
        if t.host_class == "hpc_ral_cpu" and instrument in (None, t.instrument)
    ]


def batch1_a100(instrument: str) -> list[Task]:
    return [t for t in planned_tasks() if t.host_class == "hpc_a100" and t.instrument == instrument]


def _is_bounding(task: Task) -> bool:
    return task.instrument == BOUNDING_INSTRUMENT[task.bound]


def batch0_ral_cpu() -> list[Task]:
    return [t for t in batch1_ral_cpu() if _is_bounding(t)]


def batch0_a100() -> list[Task]:
    return [t for t in planned_tasks() if t.host_class == "hpc_a100" and _is_bounding(t)]


def batch0_references() -> list[Task]:
    return reference_tasks(("jvla",), batch=0)


def basis_leg(task: Task) -> Task:
    """The batch-0 leg that bounds ``task``: same script, setup, host class and measurement on
    the bounding instrument (jvla for the sparse, numba and reference legs; sma for the dense and
    MGE legs, which exist only there)."""
    from dataclasses import replace

    return replace(task, instrument=BOUNDING_INSTRUMENT[task.bound], batch=0)


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


def peak_rss_gib(instrument: str, table: dict | None = None) -> float:
    """Batch 0's peak MaxRSS (GiB) bounding ``instrument``: the smallest measured instrument at
    or above it (instruments ordered by visibilities), else the largest measured one."""
    table = BATCH0_PEAK_RSS_KIB if table is None else table
    order = [i for i in INSTRUMENTS if i in table]
    above = [i for i in order if INSTRUMENTS.index(i) >= INSTRUMENTS.index(instrument)]
    return table[(above or order)[0 if above else -1]][0] / 2**20


def mem_gb_measured(instruments, minimum: int = BATCH1_MEM_FLOOR_GB, table=None) -> int:
    """Batch 1's ``--mem``: the next step at or above MEM_HEADROOM x the bounding peak."""
    peak = max(peak_rss_gib(i, table) for i in instruments)
    return next(step for step in MEM_STEPS_GB if step >= max(minimum, MEM_HEADROOM * peak))


# --------------------------------------------------------------------------- dataset preflight


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """Simulate ``dataset/interferometer/<instrument>`` if missing and cache its adapt image and
    NumPy W~ preload, under a lock.

    Array tasks of one instrument start together; without the lock two would simulate the
    dataset (or write ``lensed_source.fits`` / the preload ``.npy``) at once and a third could
    read a half-written file. Returns early once both caches exist.
    """
    import fcntl

    lock_dir = root / "output"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"p2b_dataset_{instrument}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sys.path.insert(0, str(root / "scripts" / "misc"))
        from instruments.interferometer import INSTRUMENTS as GEOMETRY

        dataset_path = root / "dataset" / "interferometer" / instrument
        ny, nx = GEOMETRY[instrument]["real_space_shape"]
        radius = float(GEOMETRY[instrument]["mask_radius"])
        preload = dataset_path / f"nufft_precision_operator_nufft_{ny}x{nx}_r{radius}.npy"
        if (dataset_path / "lensed_source.fits").exists() and preload.exists():
            return
        from likelihood_breakdown import interferometer_pixelized as shared
        from likelihood_breakdown import interferometer_pixelized_numpy as numpy_harness
        from likelihood_breakdown import timing

        from _adapt_image_util import adapt_image_for_dataset
        from _profile_cli import auto_simulate_if_missing

        auto_simulate_if_missing(
            dataset_path, dataset_type="interferometer", instrument=instrument, workspace_root=root
        )
        timer = timing.Timer()
        ds = shared.load_dataset(instrument, None, root, timer)
        adapt_image_for_dataset(dataset_path=ds.dataset_path, dataset=ds.dataset)
        numpy_harness.preload_sparse_dataset(ds, 128, True, timer)


# --------------------------------------------------------------------------- submit rendering


def _campaign(*, matrixfree: bool = False) -> _common.Campaign:
    if matrixfree:
        return _common.Campaign(
            pin_var="P2B_PIN",
            pin="main-unreleased",
            hpc_base=HPC_BASE,
            private_base=MF_PRIVATE_BASE,
            logs="p2b_logs",
            prefix="p2b",
            module="interferometer_p2b",
            allow_var="P2B_ALLOW_PROVISIONAL_WALL",
            private_var="MF_PRIVATE_BASE",
            shas=MF_LIBRARY_SHAS,
        )
    return _common.Campaign(
        pin_var="P2B_PIN",
        pin=PIN,
        hpc_base=HPC_BASE,
        private_base=PRIVATE_BASE,
        logs="p2b_logs",
        prefix="p2b",
        module="interferometer_p2b",
        allow_var="P2B_ALLOW_PROVISIONAL_WALL",
    )


def _mem(tasks: list[Task], device: str, batch: int) -> tuple[int, str]:
    """(``--mem`` GB, the header line saying where it came from)."""
    instruments = sorted({t.instrument for t in tasks}, key=INSTRUMENTS.index)
    if tasks[0].kind == "reference":
        if batch == 1 and WALL_BASIS_STATUS == "measured":
            mem = mem_gb_measured(instruments, table=REFERENCE_PEAK_RSS_KIB)
            return mem, f"# MEMORY. --mem={mem}gb from batch 0's reference MaxRSS x {MEM_HEADROOM}."
        return REFERENCE_MEM_GB, (
            f"# MEMORY. --mem={REFERENCE_MEM_GB}gb, a declared estimate: S^2 + one 5e5-visibility"
            "\n# chunk of T (~25 GB peak at S = 1280)."
        )
    if tasks[0].kind == "matrixfree":
        return MF_MEM_GB, (
            f"# MEMORY. --mem={MF_MEM_GB}gb, a declared estimate: the dense fft-route reference at"
            "\n# 192x192 holds F and H of ~11 GB each, and jvla's mapping matrix and preload."
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
        f"# MEMORY. --mem={mem}gb, a declared estimate for {', '.join(instruments)} (survey A8;"
        "\n# the largest instrument decides). Batch 0's sacct MaxRSS replaces it before batch 1."
    )


def render(name: str, tasks: list[Task], *, kind: str, batch: int, title: str) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) / ``a100_job`` / ``a100_array``."""
    first = tasks[0]
    matrixfree = first.kind == "matrixfree"
    reference = first.kind == "reference"
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    budget = "reference" if reference else device
    sequential = kind == "a100_job"
    measured = batch == 1 and WALL_BASIS_STATUS == "measured" and not matrixfree
    if matrixfree:
        guess, time = MF_WALL_GUESS_S, MF_TIME
    elif batch == 0:
        guess, time = BATCH0_WALL_GUESS_S[budget], BATCH0_TIME[budget]
    elif measured:
        guess, time = (
            None,
            slurm_time(_common.batch_wall_s(tasks, WALL_BASIS, sequential=sequential)),
        )
    else:
        guess, time = None, BATCH1_PROVISIONAL_TIME[budget]
    results_root = (
        "$AP_ROOT/output/p2b_matrixfree" if matrixfree else f"$AP_ROOT/output/p2b_batch{batch}"
    )
    mem, mem_line = _mem(tasks, device, batch)
    n = len(tasks)
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        (
            f"# Likelihood-lifecycle phase 2b interferometer campaign ({ISSUE}), matrix-free "
            f"experiment: {n} leg(s)."
            if matrixfree
            else f"# Likelihood-lifecycle phase 2b interferometer campaign ({ISSUE}), batch "
            f"{batch}: {n} leg(s)."
        ),
        "# GENERATED by scripts/misc/campaigns/interferometer_p2b.py --write; edit the generator,",
        "# not this file (the campaign test fails on a hand edit). The batch procedure is in",
        "# scripts/misc/campaigns/README.md.",
        "#",
    ]
    if matrixfree:
        header += [
            "# LIBRARY CODE. UNRELEASED library main: a private checkout ($MF_PRIVATE_BASE, default",
            f"# {MF_PRIVATE_BASE}) whose five HEADs must be the declared SHAs",
            "# (interferometer_p2b.MF_LIBRARY_SHAS; PyAutoArray#658 matrix-free route) is prepended",
            f"# to PYTHONPATH over the shared venv ($PYAUTO_HPC_BASE, default {HPC_BASE}). The job",
            "# refuses to run on any other HEAD or import origin, and checks each row's",
            "# library_revisions after its leg. The shared mirror is never pulled or used.",
            "#",
            "# ROWS. Experiments only, never current: 35x35 rows go to the cell's",
            "# experiments/numpy_cpu_matrixfree/ folder, the alma mesh scan to",
            "# scans/numpy_cpu_matrixfree_<N>x<N>/, under the gitignored output/p2b_matrixfree/ tree.",
        ]
    else:
        header += [
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
                "# REFERENCES. Each task computes the visibility-chunked eager NumPy dense log evidence",
                "# of one likelihood x instrument (ruling R6) into output/p2b_references/ (reused when",
                f"# valid), then rewrites parity.json of every cell row under output/p2b_batch{batch}/",
                "# that carries a parity_candidate.",
            ]
        else:
            header += [
                "# CELLS. Every pixelized leg runs the standard-size interferometer production cell",
                "# (--standard-size, ruling R3: Delaunay 1250 Hilbert + 30 zeroed edge points with",
                "# AdaptSplit, rectangular 35x35 RectangularBilinearAdaptImage with Adapt, the truth",
                "# lensed-source adapt image uncapped; MGE-20 unchanged) and writes its row, fit.png and",
                "# parity.json into the mirrored cell named by --output-path under the gitignored",
                f"# output/p2b_batch{batch}/ scratch tree, never in place over committed rows.",
            ]
    header += ["#", mem_line]
    if kind == "ral_cpu":
        header += [
            "# One leg per array task on `ral` (CPU work never goes on `gpu`). numba legs pin their",
            "# BLAS and numba threads to 1; JAX CPU legs get NPROC=8.",
        ]
    elif kind == "a100_job":
        header += ["# One A100 job runs this instrument's hpc_a100 cells in turn."]
    else:
        header += [
            "# One leg per array task, each with --gres=gpu:1, throttled to %2 so the timing arm",
            "# holds at most two A100s.",
        ]
    header += ["#"]
    if matrixfree:
        header += [
            f"# WALL-BASIS: nothing measured on this route yet; {guess} s per leg is a declared guess",
            "# (3x floor).",
        ]
    elif batch == 0:
        header += [
            "# BATCH 0 = TIMING ARM. It runs the legs that bound every other instrument (jvla for the",
            "# sparse, numba and reference legs; sma for the dense and MGE legs, which exist only",
            "# there). Each leg's wall lands in output/p2b_logs/walls.tsv; batch 1's WALL-BASIS rows",
            "# cite them and its --time is set from them.",
            "#",
            f"# WALL-BASIS: nothing measured at the standard size yet; {guess} s per leg is a declared",
            "# guess (3x floor), and the jvla legs may run well past it. The timing arm exists to",
            "# find out.",
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
            "# WALL_BASIS_STATUS=provisional (export P2B_ALLOW_PROVISIONAL_WALL=1 overrides it).",
        ]
    header += _common.wall_rows(
        tasks, device, guess, measured=measured, sequential=sequential, basis=WALL_BASIS
    )
    job = "p2b_" + name.split("submit_campaign_p2b_", 1)[1].removesuffix("_fp64")
    sbatch = _common.sbatch_block(
        job,
        kind,
        n,
        mem=mem,
        time=time,
        throttle=("%8" if batch == 0 else "%16") if kind == "ral_cpu" else None,
    )
    campaign = _campaign(matrixfree=matrixfree)
    guard = (
        _common.provisional_guard(WALL_BASIS_STATUS, campaign.allow_var)
        if batch == 1 and not matrixfree
        else None
    )
    return _common.assemble(
        header,
        sbatch,
        guard,
        _common.preamble(campaign, name),
        _common.manifest_arrays(tasks, results_root, module="interferometer_p2b"),
        _common.run_block(kind),
    )


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text}."""
    out = {}

    def add(folder, name, tasks, **kwargs):
        out[f"hpc/{folder}/{name}"] = render(name, tasks, **kwargs)

    add(
        "batch_cpu",
        "submit_campaign_p2b_batch0_ral_cpu_fp64",
        batch0_ral_cpu(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm, RAL CPU: the bounding hpc_ral_cpu interferometer cells.",
    )
    add(
        "batch_gpu",
        "submit_campaign_p2b_batch0_a100_fp64",
        batch0_a100(),
        kind="a100_array",
        batch=0,
        title="Batch-0 timing arm, A100: the bounding hpc_a100 interferometer cells.",
    )
    add(
        "batch_cpu",
        "submit_campaign_p2b_batch0_references_ral_cpu_fp64",
        batch0_references(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm (optional), RAL CPU: the jvla chunked dense parity references.",
    )
    for instrument in INSTRUMENTS:
        add(
            "batch_cpu",
            f"submit_campaign_p2b_interferometer_ral_cpu_{instrument}_fp64",
            batch1_ral_cpu(instrument),
            kind="ral_cpu",
            batch=1,
            title=f"RAL CPU array: the hpc_ral_cpu interferometer cells of {instrument}.",
        )
        add(
            "batch_gpu",
            f"submit_campaign_p2b_interferometer_a100_{instrument}_fp64",
            batch1_a100(instrument),
            kind="a100_job",
            batch=1,
            title=f"A100 job: the hpc_a100 interferometer cells of {instrument}.",
        )
    add(
        "batch_cpu",
        "submit_campaign_p2b_references_ral_cpu_fp64",
        reference_tasks(),
        kind="ral_cpu",
        batch=1,
        title="RAL CPU array: the chunked dense parity references above sma (alma, alma_high, jvla).",
    )
    add(
        "batch_cpu",
        "submit_campaign_p2b_matrixfree_ral_cpu_fp64",
        matrixfree_tasks(),
        kind="ral_cpu",
        batch=1,
        title="RAL CPU array: the matrix-free rectangular experiment (35x35 everywhere, alma scan).",
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
        for task in planned_tasks() + reference_tasks() + matrixfree_tasks():
            print(f"{task.cell:<78} {task.script}.py {task.flags}")
        return 0
    return _common.write_or_check(submits(), ROOT, args=args, generator="interferometer_p2b.py")


if __name__ == "__main__":
    sys.exit(main())
