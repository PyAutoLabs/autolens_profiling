"""Likelihood-lifecycle phase 2a: the imaging campaign's task manifest and its SLURM submits.

autolens_profiling#456 re-measures the 106 planned imaging cells (``imaging/{mge, rectangular,
delaunay, delaunay_nn}``; ``dashboard/coverage.json``) at the standard size on the pinned release
2026.10.23.1, from a private RAL checkout. This module is the one copy of that work list: every
cell becomes one :class:`Task` (script, flags, output path), and the submits under ``hpc/`` are
rendered from the tasks, so the manifest the tests check and the bash arrays the jobs run cannot
drift apart::

    python scripts/misc/campaigns/imaging_p2a.py --write   # (re)render the submits
    python scripts/misc/campaigns/imaging_p2a.py --check   # exit 1 if a submit is stale
    python scripts/misc/campaigns/imaging_p2a.py --list    # print the manifest

The submits (README.md beside this file has the batch procedure):

- batch 1, ``hpc/batch_cpu/submit_campaign_p2a_imaging_ral_cpu_fp64`` — one ``--partition=ral``
  array over the 44 ``hpc_ral_cpu`` cells, one cell per task;
- batch 1, ``hpc/batch_gpu/submit_campaign_p2a_imaging_a100_<instrument>_fp64`` — one
  ``--gres=gpu:1`` job per instrument running that instrument's ``hpc_a100`` cells in turn,
  including the ``jax_cpu_sparse_fp64@hpc_a100`` legs, which run on the GPU node's CPU inside
  the GPU job (never a CPU array on the ``gpu`` partition);
- batch 0, ``..._batch0_ral_cpu_fp64`` and ``..._batch0_a100_fp64`` — the timing arms that set
  batch 1's ``--time`` (:data:`BATCH0_INSTRUMENTS`), writing their rows into the gitignored
  ``output/p2a_batch0/`` scratch tree;
- the 28x28 rectangular size variant (autolens_profiling#468, :func:`rect28_tasks`),
  ``hpc/batch_cpu/submit_campaign_p2a_rect28_ral_cpu_fp64`` (``--partition=ral`` array) and
  ``hpc/batch_gpu/submit_campaign_p2a_rect28_a100_fp64`` (one ``--gres=gpu:1`` job): hst
  rectangular at 784 source pixels, the mesh the SLaM inference runs use, on tag
  :data:`RECT28_PIN`, writing into ``output/p2a_rect28/``.

Stdlib only; nothing here imports the modelling stack except :func:`prepare_dataset`, which the
submits call to simulate a missing dataset and cache its adapt image under a lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _script_routes import load_routes  # noqa: E402

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

#: The release every cell is measured on (human ruling 2026-10-24), and where the private RAL
#: checkout of the five libraries at that tag lives (override with ``PYAUTO_PRIVATE_BASE``).
PIN = "2026.10.23.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
#: The shared virtualenv (third-party packages only; hpc_ral.md, ``activate.sh``).
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#456"

INSTRUMENTS = ("ao", "euclid", "hst", "jwst")
VARIANT_INSTRUMENTS = {
    "mge": INSTRUMENTS,
    "rectangular": INSTRUMENTS,
    "delaunay": INSTRUMENTS,
    "delaunay_nn": ("hst",),
}
PIXELIZED = ("rectangular", "delaunay", "delaunay_nn")
#: §3.3 provisional designations (``catalogue/provisional_designations.json``), optimal and
#: secondary, per host class.
SETUPS = {
    "parametric": {
        "hpc_ral_cpu": ("jax_cpu_dense_fp64",),
        "hpc_a100": ("jax_gpu_dense_fp64",),
    },
    "pixelized": {
        "hpc_ral_cpu": ("numba_cpu_sparse_fp64", "jax_cpu_sparse_fp64"),
        "hpc_a100": ("jax_gpu_dense_fp64", "jax_gpu_sparse_fp64", "jax_cpu_sparse_fp64"),
    },
}
MEASUREMENTS = ("runtime", "breakdown")
#: ``--config-name`` per (host class, device): the host tier a row's witness judges against.
CONFIG_NAMES = {
    ("hpc_ral_cpu", "cpu"): "hpc_ral_cpu_fp64",
    ("hpc_a100", "gpu"): "hpc_a100_fp64",
    ("hpc_a100", "cpu"): "hpc_a100_cpu_fp64",
}

#: Batch 0 measures the instrument that bounds every other one (masked pixels at the 3.5" mask
#: scale as 1 / pixel_scale^2: euclid ~3.8k < hst ~15k < jwst ~43k < ao ~385k), plus hst for the
#: delaunay_nn cells, which exist only there. A cell's wall on ao bounds the same script on the
#: smaller instruments, so batch 1's ``--time`` rests on a measurement, not a guess.
BATCH0_INSTRUMENTS = {
    "mge": ("ao",),
    "rectangular": ("ao",),
    "delaunay": ("ao",),
    "delaunay_nn": ("hst",),
}
#: Batch 0's declared per-task wall guess (``source: unmeasured``, 3x floor) and its ``--time``.
BATCH0_WALL_GUESS_S = 14000
BATCH0_TIME = "12:00:00"

#: Batch 0's measured legs, on tag 2026.10.23.1: RAL CPU array 412877, A100 array 412884, and the
#: breakdown legs re-run after PR #458 as 412911 (CPU tasks 0, 4, 8, 10) and 412915 (A100 tasks
#: 0, 4, 6, 10, 12, 14, 16, 18). Per leg: the ``walls.tsv`` WALL (the cell script alone), the
#: sacct Elapsed of its array task (preamble, dataset preparation, script and revision check),
#: and that task. Batch 1's walls rest on the larger of the two.
BATCH0_WALLS = {
    "imaging/delaunay/ao jax_cpu_sparse_fp64@hpc_ral_cpu [breakdown]": (1326, 1330, "412911_0"),
    "imaging/delaunay/ao jax_cpu_sparse_fp64@hpc_ral_cpu [runtime]": (1338, 1428, "412877_1"),
    "imaging/delaunay/ao numba_cpu_sparse_fp64@hpc_ral_cpu [breakdown]": (1329, 1419, "412877_2"),
    "imaging/delaunay/ao numba_cpu_sparse_fp64@hpc_ral_cpu [runtime]": (1877, 1967, "412877_3"),
    "imaging/delaunay_nn/hst jax_cpu_sparse_fp64@hpc_ral_cpu [breakdown]": (212, 216, "412911_4"),
    "imaging/delaunay_nn/hst jax_cpu_sparse_fp64@hpc_ral_cpu [runtime]": (308, 357, "412877_5"),
    "imaging/delaunay_nn/hst numba_cpu_sparse_fp64@hpc_ral_cpu [breakdown]": (81, 84, "412877_6"),
    "imaging/delaunay_nn/hst numba_cpu_sparse_fp64@hpc_ral_cpu [runtime]": (67, 69, "412877_7"),
    "imaging/mge/ao jax_cpu_dense_fp64@hpc_ral_cpu [breakdown]": (36, 40, "412911_8"),
    "imaging/mge/ao jax_cpu_dense_fp64@hpc_ral_cpu [runtime]": (125, 128, "412877_9"),
    "imaging/rectangular/ao jax_cpu_sparse_fp64@hpc_ral_cpu [breakdown]": (1142, 1145, "412911_10"),
    "imaging/rectangular/ao jax_cpu_sparse_fp64@hpc_ral_cpu [runtime]": (1172, 1174, "412877_11"),
    "imaging/rectangular/ao numba_cpu_sparse_fp64@hpc_ral_cpu [breakdown]": (
        1375,
        1377,
        "412877_12",
    ),
    "imaging/rectangular/ao numba_cpu_sparse_fp64@hpc_ral_cpu [runtime]": (1836, 1838, "412877_13"),
    "imaging/delaunay/ao jax_cpu_sparse_fp64@hpc_a100 [breakdown]": (1523, 1533, "412915_0"),
    "imaging/delaunay/ao jax_cpu_sparse_fp64@hpc_a100 [runtime]": (1432, 1522, "412884_1"),
    "imaging/delaunay/ao jax_gpu_dense_fp64@hpc_a100 [breakdown]": (103, 107, "412884_2"),
    "imaging/delaunay/ao jax_gpu_dense_fp64@hpc_a100 [runtime]": (210, 214, "412884_3"),
    "imaging/delaunay/ao jax_gpu_sparse_fp64@hpc_a100 [breakdown]": (121, 131, "412915_4"),
    "imaging/delaunay/ao jax_gpu_sparse_fp64@hpc_a100 [runtime]": (187, 191, "412884_5"),
    "imaging/delaunay_nn/hst jax_cpu_sparse_fp64@hpc_a100 [breakdown]": (225, 228, "412915_6"),
    "imaging/delaunay_nn/hst jax_cpu_sparse_fp64@hpc_a100 [runtime]": (329, 334, "412884_7"),
    "imaging/delaunay_nn/hst jax_gpu_dense_fp64@hpc_a100 [breakdown]": (59, 63, "412884_8"),
    "imaging/delaunay_nn/hst jax_gpu_dense_fp64@hpc_a100 [runtime]": (94, 97, "412884_9"),
    "imaging/delaunay_nn/hst jax_gpu_sparse_fp64@hpc_a100 [breakdown]": (99, 102, "412915_10"),
    "imaging/delaunay_nn/hst jax_gpu_sparse_fp64@hpc_a100 [runtime]": (318, 322, "412884_11"),
    "imaging/mge/ao jax_gpu_dense_fp64@hpc_a100 [breakdown]": (35, 38, "412915_12"),
    "imaging/mge/ao jax_gpu_dense_fp64@hpc_a100 [runtime]": (85, 88, "412884_13"),
    "imaging/rectangular/ao jax_cpu_sparse_fp64@hpc_a100 [breakdown]": (1324, 1327, "412915_14"),
    "imaging/rectangular/ao jax_cpu_sparse_fp64@hpc_a100 [runtime]": (1264, 1268, "412884_15"),
    "imaging/rectangular/ao jax_gpu_dense_fp64@hpc_a100 [breakdown]": (74, 77, "412915_16"),
    "imaging/rectangular/ao jax_gpu_dense_fp64@hpc_a100 [runtime]": (174, 178, "412884_17"),
    "imaging/rectangular/ao jax_gpu_sparse_fp64@hpc_a100 [breakdown]": (99, 103, "412915_18"),
    "imaging/rectangular/ao jax_gpu_sparse_fp64@hpc_a100 [runtime]": (168, 171, "412884_19"),
}
#: Batch-0 legs that cannot run, by human ruling a known limitation rather than a bug. Batch 1
#: keeps them (each fails in under two minutes and the job carries on); their wall above is that
#: failure, so the A100 job's ``--time`` still holds them. Empty since autolens_profiling#466:
#: the two ao ``jax_gpu_dense_fp64@hpc_a100 [breakdown]`` legs ran out of A100 memory only
#: because the breakdown held duplicate device copies of the mapping matrices; freed, both run
#: (RAL 413033, 131 s and 111 s).
BATCH0_LIMITATIONS: dict = {}
#: ``--time`` = WALL_HEADROOM x the measured wall (README "Setting batch 1's --time"), rounded
#: up to the next half hour.
WALL_HEADROOM = 1.25
WALL_ROUND_S = 1800
#: Slack kept above WALL_HEADROOM x the wall after rounding: an A100 job that overruns loses
#: every leg still to run, so a rounding that lands just above 1.25x is pushed to the next step.
WALL_MIN_SLACK_S = 900
#: Batch 1's smallest ``--mem``: the hst and euclid rectangular / delaunay legs were never
#: measured on their own instrument, and an out-of-memory kill loses the leg.
BATCH1_MEM_FLOOR_GB = 32
#: ``measured`` once the batch-1 WALL-BASIS rows cite batch 0 (README "Setting batch 1's
#: --time"); while ``provisional``, batch 1 renders unmeasured rows, the placeholder ``--time``
#: below, and a submit that refuses to run.
WALL_BASIS_STATUS = "measured"
BATCH1_PROVISIONAL_TIME = {"ral_cpu": "12:00:00", "a100": "2-00:00:00"}

#: The 28x28 rectangular size variant (autolens_profiling#468, human 2026-10-26): hst
#: rectangular at the 784 source pixels (``catalogue/size_variants.json``) the SLaM inference
#: runs use, beside the 35x35 standard, so their wall-time rows pair with a profiling row of the
#: same size. Measured on tag 2026.10.25.1 from its own private checkout. Each setup is one the
#: inference runs record, on the host class it ran on (light_fixed and jax_cpu_dense have no
#: profiling cell at any size).
RECT28_PIN = "2026.10.25.1"
RECT28_PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{RECT28_PIN}"
RECT28_ISSUE = "autolens_profiling#468"
RECT28_SOURCE_PIXELS = 784
RECT28_INSTRUMENT = "hst"
#: Phase-3 day-zero JAX re-measure (human 2026-10-30, Q7: day zero is one environment): every
#: JAX cell of the batch rerun on tag 2026.10.25.1 under the shared venv's jax/jaxlib 0.11.2, with
#: the shared preamble's --xla_cpu_multi_thread_eigen=false on CPU legs (D12). The 2a rows ran on
#: 2026.10.23.1 with jax 0.10.2, which the tag's PyAutoNerves excludes; numba rows are carried.
JAX_RERUN_PIN = "2026.10.25.1"
JAX_RERUN_PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{JAX_RERUN_PIN}"
JAX_RERUN_ISSUE = "likelihood-lifecycle phase 3, Q7"
RECT28_SETUPS = (
    ("numba_cpu_sparse_fp64", "hpc_ral_cpu"),
    ("jax_cpu_sparse_fp64", "hpc_a100"),
    ("jax_gpu_sparse_fp64", "hpc_a100"),
    ("jax_gpu_dense_fp64", "hpc_a100"),
)

#: Pixel scale per instrument (arcsec / pixel; ``scripts/misc/simulators/imaging.INSTRUMENTS``,
#: pinned equal by the campaign test) and the cells' 3.5" circular mask: the masked image size
#: every cell's memory scales with.
PIXEL_SCALES = {"ao": 0.01, "euclid": 0.1, "hst": 0.05, "jwst": 0.03}
MASK_RADIUS = 3.5
#: Columns of the dense mapping matrix: 1280 Delaunay vertices (rectangular 1225) plus up to 60
#: linear MGE Gaussians.
MAPPING_COLUMNS = 1340
MEM_STEPS_GB = (16, 32, 64, 128, 256)


def masked_pixels(instrument: str) -> int:
    """Image pixels inside the 3.5" mask: pi r^2 / pixel_scale^2 (ao ~385k, euclid ~3.8k)."""
    import math

    return int(math.pi * MASK_RADIUS**2 / PIXEL_SCALES[instrument] ** 2)


def mem_gb(instruments, minimum: int = 16) -> int:
    """``--mem`` for a job over ``instruments``: the largest instrument decides.

    Estimated peak = 4 GB (Python, JAX, numba, libraries) + eight fp64 arrays the size of the
    dense mapping matrix (masked pixels x 1340 columns): the mapping and blurred mapping
    matrices of the cell, the same two of the eager NumPy dense parity reference, and JAX /
    numba working copies. The request is the next step of 16 / 32 / 64 / 128 / 256 GB at or above
    twice that peak. An estimate from the dataset sizes, not a measurement: batch 0's
    ``sacct --format=MaxRSS`` replaces it before batch 1.
    """
    peak = max(4.0 + 8 * masked_pixels(i) * MAPPING_COLUMNS * 8 / 1e9 for i in instruments)
    return next(step for step in MEM_STEPS_GB if step >= max(minimum, 2 * peak))


#: Batch 0's peak resident memory (sacct MaxRSS, KiB) on the two instruments it ran, the largest
#: of any leg there (CPU array or A100 job), and the task that reached it.
BATCH0_PEAK_RSS_KIB = {"hst": (10652884, "412877_5"), "ao": (36762308, "412877_1")}
#: Batch 1's ``--mem`` is at least MEM_HEADROOM x the measured (or interpolated) peak.
MEM_HEADROOM = 1.5


def peak_rss_gib(instrument: str) -> float:
    """Batch 0's peak MaxRSS (GiB) at ``instrument``'s dataset size.

    Measured on hst (15k masked pixels) and ao (385k); jwst is interpolated linearly in masked
    pixels between them, and euclid (below hst) takes the hst peak: the peak is mostly a fixed
    ~9 GiB plus a part that grows with the image, and nothing smaller than hst was measured.
    """
    (lo, (lo_kib, _)), (hi, (hi_kib, _)) = sorted(
        BATCH0_PEAK_RSS_KIB.items(), key=lambda kv: masked_pixels(kv[0])
    )
    lo_gib, hi_gib = lo_kib / 2**20, hi_kib / 2**20
    slope = (hi_gib - lo_gib) / (masked_pixels(hi) - masked_pixels(lo))
    return lo_gib + slope * max(0, masked_pixels(instrument) - masked_pixels(lo))


def mem_gb_measured(instruments, minimum: int = 16) -> int:
    """Batch 1's ``--mem``: the next 16 / 32 / 64 / 128 / 256 GB step at or above MEM_HEADROOM x
    the largest instrument's :func:`peak_rss_gib` (SLURM's ``gb`` is GiB)."""
    peak = max(peak_rss_gib(i) for i in instruments)
    return next(step for step in MEM_STEPS_GB if step >= max(minimum, MEM_HEADROOM * peak))


_ROUTES = {row["path"]: row["legacy"] for row in load_routes(ROOT)["routes"]}


@dataclass(frozen=True, order=True)
class Task:
    """One planned cell: ``imaging/<variant>/<instrument> <setup_id>@<host_class> [<measurement>]``.

    ``source_pixels`` is a declared size variant (``catalogue/size_variants.json``) the cell is
    built at instead of the standard size, else None.
    """

    variant: str
    instrument: str
    setup_id: str
    host_class: str
    measurement: str
    source_pixels: int | None = None

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
        """The coverage-matrix cell name (``dashboard/coverage.json`` ``cells[].cell``)."""
        name = (
            f"imaging/{self.variant}/{self.instrument} {self.setup_id}@{self.host_class} "
            f"[{self.measurement}]"
        )
        return name + (f" {self.size_folder}" if self.source_pixels else "")

    @property
    def size_folder(self) -> str | None:
        """The cell sub-folder a size variant's row lives in (``source_pixels_<n>``)."""
        return f"source_pixels_{self.source_pixels}" if self.source_pixels else None

    @property
    def script(self) -> str:
        """The cell script, relative to ``scripts/`` and without ``.py``."""
        if self.backend == "numba":
            folder = "rectangular" if self.variant == "rectangular" else "delaunay"
            return f"imaging/{folder}/likelihood_{self.measurement}_numba"
        return f"imaging/{self.variant}/likelihood_{self.measurement}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id: ``imaging/<legacy script stem>`` (``wall/check_submits.py``)."""
        legacy = _ROUTES[f"scripts/{self.script}.py"]
        return f"imaging/{Path(legacy).stem}"

    @property
    def config_name(self) -> str:
        return CONFIG_NAMES[(self.host_class, self.device)]

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (the submits add those)."""
        flags = []
        if self.variant in PIXELIZED:
            flags.append("--standard-size")
            if self.source_pixels:
                flags += ["--source-pixels", str(self.source_pixels)]
        if self.backend == "jax" and self.operator == "sparse":
            flags.append("--sparse")
        if self.backend == "numba" and self.variant == "delaunay_nn":
            flags += ["--mesh-class", "delaunay_nn"]
        flags += ["--config-name", self.config_name]
        return " ".join(flags)

    @property
    def output(self) -> str:
        """The row's mirrored-tree path, relative to the results root."""
        size = f"{self.size_folder}/" if self.source_pixels else ""
        return (
            f"results/imaging/{self.variant}/{self.instrument}/{self.setup_id}/"
            f"{self.host_class}/{size}{self.measurement}.json"
        )

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS`` for the leg (a numba cell never touches a JAX device)."""
        return "cuda,cpu" if self.device == "gpu" else "cpu"


def planned_tasks() -> list[Task]:
    """Every planned imaging cell of the campaign, sorted (106: 44 RAL CPU + 62 A100)."""
    tasks = []
    for variant, instruments in VARIANT_INSTRUMENTS.items():
        table = SETUPS["pixelized" if variant in PIXELIZED else "parametric"]
        for instrument in instruments:
            for host_class, setups in table.items():
                for setup_id in setups:
                    for measurement in MEASUREMENTS:
                        tasks.append(Task(variant, instrument, setup_id, host_class, measurement))
    return sorted(tasks)


def batch1_ral_cpu(instrument: str | None = None) -> list[Task]:
    return [
        t
        for t in planned_tasks()
        if t.host_class == "hpc_ral_cpu" and instrument in (None, t.instrument)
    ]


def batch1_a100(instrument: str) -> list[Task]:
    return [t for t in planned_tasks() if t.host_class == "hpc_a100" and t.instrument == instrument]


def rect28_tasks(host_class: str | None = None) -> list[Task]:
    """The 28x28 size-variant legs (8: numba on ``hpc_ral_cpu``, six on ``hpc_a100``), sorted."""
    return sorted(
        Task("rectangular", RECT28_INSTRUMENT, sid, host, m, RECT28_SOURCE_PIXELS)
        for sid, host in RECT28_SETUPS
        for m in MEASUREMENTS
        if host_class in (None, host)
    )


def jax_rerun_tasks(instrument: str, host_class: str) -> list[Task]:
    """The JAX cells of ``instrument`` on ``host_class``: the phase-3 re-measure on jax 0.11.2."""
    tasks = batch1_ral_cpu(instrument) if host_class == "hpc_ral_cpu" else batch1_a100(instrument)
    return [t for t in tasks if t.setup_id.startswith("jax_")]


def _batch0(tasks: list[Task]) -> list[Task]:
    return [t for t in tasks if t.instrument in BATCH0_INSTRUMENTS[t.variant]]


def batch0_ral_cpu() -> list[Task]:
    return _batch0(batch1_ral_cpu())


def batch0_a100() -> list[Task]:
    return _batch0([t for t in planned_tasks() if t.host_class == "hpc_a100"])


def basis_leg(task: Task) -> Task:
    """The batch-0 leg that bounds ``task``: same script, setup, host class and measurement on
    the bounding instrument (ao; hst for delaunay_nn), at the standard size. A wall on ao bounds
    every smaller instrument, and the 35x35 standard bounds the smaller 28x28 size variant."""
    instrument = BATCH0_INSTRUMENTS[task.variant][0]
    return Task(task.variant, instrument, task.setup_id, task.host_class, task.measurement)


def measured_wall(task: Task) -> tuple[int, str]:
    """(seconds, batch-0 task) for ``task``: the larger of its basis leg's WALL and Elapsed."""
    wall, elapsed, ref = BATCH0_WALLS[basis_leg(task).cell]
    return max(wall, elapsed), ref


def batch1_wall_s(tasks: list[Task], *, sequential: bool) -> int:
    """The measured wall a batch-1 submit must hold: the slowest leg of an array (one leg per
    task), or the sum of the legs of a job that runs them in turn."""
    return _common.batch_wall_s(tasks, WALL_BASIS, sequential=sequential)


def slurm_time(seconds: float) -> str:
    """``H:MM:SS`` at WALL_HEADROOM x ``seconds`` plus WALL_MIN_SLACK_S, rounded up to the next
    WALL_ROUND_S."""
    return _common.slurm_time(
        seconds, headroom=WALL_HEADROOM, slack_s=WALL_MIN_SLACK_S, round_s=WALL_ROUND_S
    )


#: How batch 1 cites batch 0 (``_common.wall_rows``).
WALL_BASIS = _common.WallBasis(
    measured_wall=measured_wall,
    basis_leg=basis_leg,
    limitations=BATCH0_LIMITATIONS,
    walls=BATCH0_WALLS,
    headroom=WALL_HEADROOM,
)


# --------------------------------------------------------------------------- dataset preflight


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """Simulate ``dataset/imaging/<instrument>`` if missing and cache its adapt image, under a lock.

    Array tasks of one instrument start together; without the lock two of them would simulate
    the dataset (or write ``lensed_source.fits``) at once and a third could read a half-written
    file. The adapt image is built the way the JAX cells build theirs (3.5" circular mask,
    light-profile bins [4, 2, 2] at [0.3, 0.6]) so the cache does not depend on which cell
    happens to run first.
    """
    import fcntl

    lock_dir = root / "output"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"p2a_dataset_{instrument}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        dataset_path = root / "dataset" / "imaging" / instrument
        if (dataset_path / "lensed_source.fits").exists():
            return
        sys.path.insert(0, str(root / "scripts" / "misc"))
        import autolens as al
        from simulators.imaging import INSTRUMENTS as SIMULATED

        from _adapt_image_util import adapt_image_for_dataset
        from _profile_cli import auto_simulate_if_missing

        auto_simulate_if_missing(
            dataset_path, dataset_type="imaging", instrument=instrument, workspace_root=root
        )
        dataset = al.Imaging.from_fits(
            data_path=dataset_path / "data.fits",
            psf_path=dataset_path / "psf.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            pixel_scales=SIMULATED[instrument]["pixel_scale"],
        )
        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.5
        )
        dataset = dataset.apply_mask(mask=mask)
        dataset = dataset.apply_over_sampling(
            over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
                grid=dataset.grid,
                sub_size_list=[4, 2, 2],
                radial_list=[0.3, 0.6],
                centre_list=[(0.0, 0.0)],
            ),
            over_sample_size_pixelization=1,
        )
        adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)


# --------------------------------------------------------------------------- submit rendering


def _campaign(pin: str, private_base: str) -> _common.Campaign:
    return _common.Campaign(
        pin_var="P2A_PIN",
        pin=pin,
        hpc_base=HPC_BASE,
        private_base=private_base,
        logs="p2a_logs",
        prefix="p2a",
        module="imaging_p2a",
        allow_var="P2A_ALLOW_PROVISIONAL_WALL",
    )


def render(
    name: str,
    tasks: list[Task],
    *,
    kind: str,
    batch: int,
    title: str,
    rect28: bool = False,
    jax_rerun: bool = False,
) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) / ``a100_job`` / ``a100_array``.

    ``rect28``: the 28x28 size-variant batch (:func:`rect28_tasks`), rendered as a measured
    batch 1 on its own pin, private checkout and scratch tree. ``jax_rerun``: the phase-3 JAX
    re-measure (:func:`jax_rerun_tasks`), likewise on its own pin and scratch tree.
    """
    if rect28:
        pin, private_base = RECT28_PIN, RECT28_PRIVATE_BASE
    elif jax_rerun:
        pin, private_base = JAX_RERUN_PIN, JAX_RERUN_PRIVATE_BASE
    else:
        pin, private_base = PIN, PRIVATE_BASE
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    guess = BATCH0_WALL_GUESS_S if batch == 0 else None
    sequential = kind == "a100_job"
    measured = batch == 1 and WALL_BASIS_STATUS == "measured"
    time_basis_s = batch1_wall_s(tasks, sequential=sequential) if measured else None
    if measured:
        time = slurm_time(time_basis_s)
    else:
        time = BATCH0_TIME if batch == 0 else BATCH1_PROVISIONAL_TIME[device]
    n = len(tasks)
    # Never in place over committed rows: PR B copies complete rows into results/ (README).
    tree = "p2a_rect28" if rect28 else "p3_jax_rerun" if jax_rerun else f"p2a_batch{batch}"
    results_root = f"$AP_ROOT/output/{tree}"
    instruments = sorted({t.instrument for t in tasks})
    floor = BATCH1_MEM_FLOOR_GB if batch == 1 else (16 if kind == "ral_cpu" else 32)
    mem = mem_gb_measured(instruments, floor) if batch == 1 else mem_gb(instruments, floor)
    pixels = ", ".join(f"{i} {masked_pixels(i)}" for i in instruments)
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        (
            f"# Likelihood-lifecycle phase 2a, 28x28 rectangular size variant ({RECT28_ISSUE}): "
            f"{n} cell(s)."
            if rect28
            else f"# Likelihood-lifecycle phase-3 JAX re-measure of the 2a cells ({JAX_RERUN_ISSUE}): "
            f"{n} cell(s)."
            if jax_rerun
            else f"# Likelihood-lifecycle phase 2a imaging campaign ({ISSUE}), batch {batch}: "
            f"{n} cell(s)."
        ),
        "# GENERATED by scripts/misc/campaigns/imaging_p2a.py --write; edit the generator, not this",
        "# file (the campaign test fails on a hand edit). The batch procedure is in",
        "# scripts/misc/campaigns/README.md.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at tag {pin} ($PYAUTO_PRIVATE_BASE,",
        f"# default {private_base}) is prepended to PYTHONPATH over the shared venv",
        f"# ($PYAUTO_HPC_BASE, default {HPC_BASE}, third-party packages only). The job refuses to",
        "# run if a library imports from anywhere else or sits at another tag, and after each leg it",
        "# checks the row's library_revisions equal the checkout's HEADs. The shared mirror is never",
        "# pulled or used. AP_ROOT is the checkout this was submitted from (hpc/<batch>/../..).",
        "#",
    ]
    if rect28:
        header += [
            "# CELLS. Every leg runs the standard-size production cell at the declared size variant",
            f"# --standard-size --source-pixels {RECT28_SOURCE_PIXELS} (catalogue/size_variants.json):"
            " rectangular",
            "# 28x28, the mesh the SLaM inference runs use, everything else as the 35x35 standard",
            "# cells. It writes its row, fit.png and parity.json into the cell's source_pixels_784/",
            "# folder named by --output-path under the gitignored output/p2a_rect28/ scratch tree,",
            "# never in place (complete rows are copied into results/ beside the standard row).",
            "#",
        ]
    elif jax_rerun:
        header += [
            "# CELLS. Every JAX leg of the batch-1 submit of this instrument and host, rerun on tag",
            f"# {JAX_RERUN_PIN} under jax/jaxlib 0.11.2 (the CPU legs get the shared preamble's",
            "# eigen flag), so day zero is one environment (human 2026-10-30, Q7). It writes its row,",
            "# fit.png and parity.json into the mirrored cell named by --output-path under the",
            "# gitignored output/p3_jax_rerun/ scratch tree, never in place over committed rows.",
            "#",
        ]
    else:
        header += [
            "# CELLS. Every leg runs one cell script at the standard size (--standard-size: Delaunay",
            "# 1250 Hilbert vertices, rectangular 35x35; MGE has no mesh) and writes its row, fit.png and",
            "# parity.json into the mirrored cell named by --output-path under the gitignored",
            f"# output/p2a_batch{batch}/ scratch tree, never in place over committed rows"
            + (" (PR B copies complete rows into results/)." if batch == 1 else "."),
            "#",
        ]
    if batch == 1:
        rss = ", ".join(
            f"{i} {kib / 2**20:.1f} GiB in {ref}" for i, (kib, ref) in BATCH0_PEAK_RSS_KIB.items()
        )
        header += [
            f"# MEMORY. --mem={mem}gb from batch 0's sacct MaxRSS ({rss}).",
            f"# The peak at this dataset size ({pixels} masked pixels; jwst interpolated linearly",
            f"# in masked pixels, euclid takes the hst peak) x {MEM_HEADROOM}, rounded up to",
            f"# 16/32/64/128 GB, at least {floor} GB (imaging_p2a.mem_gb_measured).",
        ]
    else:
        header += [
            f"# MEMORY. --mem={mem}gb from the dataset size ({', '.join(instruments)}; masked pixels "
            f"{pixels}):",
            "# 4 GB + eight fp64 arrays of masked pixels x 1340 mapping columns, doubled, rounded up",
            "# to 16/32/64/128 GB (imaging_p2a.mem_gb). batch 0's sacct MaxRSS replaces the estimate.",
        ]
    if kind == "ral_cpu":
        header += [
            "# One cell per array task on `ral` (CPU work never goes on `gpu`). numba cells pin their",
            "# BLAS threads to 1 (production per-worker cost); JAX CPU cells get NPROC=8.",
        ]
    elif kind == "a100_job":
        header += [
            "# One A100 job runs this instrument's hpc_a100 cells in turn. The jax_cpu_sparse legs",
            "# run on the GPU node's CPU inside this GPU job (JAX_PLATFORMS=cpu, NPROC=8) — the",
            "# hpc_a100 host class's CPU setup, never a CPU array on the gpu partition.",
        ]
    else:
        header += [
            "# One leg per array task, each with --gres=gpu:1, throttled to %2 so the timing arm",
            "# holds at most two A100s. The jax_cpu_sparse legs run on the GPU node's CPU inside",
            "# their GPU task (JAX_PLATFORMS=cpu, NPROC=8).",
        ]
    header += ["#"]
    if batch == 0:
        header += [
            "# BATCH 0 = TIMING ARM. It runs the cells that bound every other instrument's wall (ao,",
            "# the most masked pixels by far, plus hst for delaunay_nn, which exists only there).",
            "# Each leg's wall lands in output/p2a_logs/walls.tsv; batch 1's WALL-BASIS rows cite",
            "# them (source: measured-wall, ref: this job) and its --time is set from them.",
            "#",
            f"# WALL-BASIS: nothing measured at the standard size yet; {BATCH0_WALL_GUESS_S} s per leg is",
            "# a declared guess (3x floor), and the ao legs may run well past it — the timing arm",
            "# exists to find out.",
        ]
    elif measured:
        rule = (
            "the sum of its legs' walls (they run in turn)"
            if sequential
            else "its slowest leg's wall (one leg per array task)"
        )
        header += [
            "# WALL-BASIS: measured by batch 0 (RAL 412877 / 412884, breakdown re-runs 412911 / 412915).",
            "# Each leg's wall is that of its batch-0 leg of the same script, setup and host class on",
            "# the bounding instrument (ao; hst for delaunay_nn), the larger of the walls.tsv WALL and",
            "# the task's sacct Elapsed, so smaller instruments run under an ao upper bound. The walls",
            "# include the headline median (same scripts and release). Each row's wall is",
            f"# {rule}; --time is {WALL_HEADROOM}x the submit's total ({time_basis_s}s),",
            f"# plus {WALL_MIN_SLACK_S // 60} min of slack, rounded up to the next {WALL_ROUND_S // 60} min.",
        ]
        if rect28:
            header += [
                "# This 28x28 batch has no wall of its own: it takes the ao legs of the 35x35",
                "# standard (batch 0 measured no hst rectangular leg), and both the 784-pixel mesh",
                "# (< 1225) and hst (15k masked pixels, < ao's 385k) are smaller, so every wall and",
                "# the MaxRSS --mem below are upper bounds.",
            ]
    else:
        header += [
            "# WALL-BASIS: PROVISIONAL. No cell has a standard-size wall yet; batch 0 measures them.",
            "# Every row is `unmeasured` until batch 0 lands, and the job refuses to run while",
            "# WALL_BASIS_STATUS=provisional (export P2A_ALLOW_PROVISIONAL_WALL=1 overrides it).",
        ]
    header += _common.wall_rows(
        tasks, device, guess, measured=measured, sequential=sequential, basis=WALL_BASIS
    )
    job = (
        f"p2a_{name.split('p2a_', 1)[1]}"
        if rect28 or jax_rerun
        else f"p2a_{'b0' if batch == 0 else 'b1'}_{name.split('imaging_', 1)[1]}"
    )
    sbatch = _common.sbatch_block(
        job,
        kind,
        n,
        mem=mem,
        time=time,
        throttle=("%8" if batch == 0 else "%16") if kind == "ral_cpu" else None,
    )
    campaign = _campaign(pin, private_base)
    return _common.assemble(
        header,
        sbatch,
        _common.provisional_guard(WALL_BASIS_STATUS, campaign.allow_var) if batch == 1 else None,
        _common.preamble(campaign, name),
        _common.manifest_arrays(tasks, results_root, module="imaging_p2a"),
        _common.run_block(kind),
    )


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text} for batch 0 and batch 1."""
    out = {}
    for instrument in INSTRUMENTS:  # one array per instrument: --mem follows the dataset size
        name = f"submit_campaign_p2a_imaging_ral_cpu_{instrument}_fp64"
        out[f"hpc/batch_cpu/{name}"] = render(
            name,
            batch1_ral_cpu(instrument),
            kind="ral_cpu",
            batch=1,
            title=f"RAL CPU array: the hpc_ral_cpu imaging cells of {instrument} at the standard size.",
        )
    for instrument in INSTRUMENTS:
        name = f"submit_campaign_p2a_imaging_a100_{instrument}_fp64"
        out[f"hpc/batch_gpu/{name}"] = render(
            name,
            batch1_a100(instrument),
            kind="a100_job",
            batch=1,
            title=f"A100 job: the hpc_a100 imaging cells of {instrument} at the standard size.",
        )
    name = "submit_campaign_p2a_imaging_batch0_ral_cpu_fp64"
    out[f"hpc/batch_cpu/{name}"] = render(
        name,
        batch0_ral_cpu(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm, RAL CPU: the bounding hpc_ral_cpu cells.",
    )
    name = "submit_campaign_p2a_imaging_batch0_a100_fp64"
    out[f"hpc/batch_gpu/{name}"] = render(
        name,
        batch0_a100(),
        kind="a100_array",
        batch=0,
        title="Batch-0 timing arm, A100: the bounding hpc_a100 cells.",
    )
    for instrument in INSTRUMENTS:
        for host_class, kind, folder, device in (
            ("hpc_ral_cpu", "ral_cpu", "batch_cpu", "ral_cpu"),
            ("hpc_a100", "a100_job", "batch_gpu", "a100"),
        ):
            name = f"submit_campaign_p2a_jaxrerun_{device}_{instrument}_fp64"
            out[f"hpc/{folder}/{name}"] = render(
                name,
                jax_rerun_tasks(instrument, host_class),
                kind=kind,
                batch=1,
                jax_rerun=True,
                title=f"Phase-3 JAX re-measure: the {host_class} JAX imaging cells of {instrument}.",
            )
    name = "submit_campaign_p2a_rect28_ral_cpu_fp64"
    out[f"hpc/batch_cpu/{name}"] = render(
        name,
        rect28_tasks("hpc_ral_cpu"),
        kind="ral_cpu",
        batch=1,
        rect28=True,
        title="RAL CPU array: the hpc_ral_cpu hst rectangular cells at 28x28 (784 source pixels).",
    )
    name = "submit_campaign_p2a_rect28_a100_fp64"
    out[f"hpc/batch_gpu/{name}"] = render(
        name,
        rect28_tasks("hpc_a100"),
        kind="a100_job",
        batch=1,
        rect28=True,
        title="A100 job: the hpc_a100 hst rectangular cells at 28x28 (784 source pixels).",
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
        for task in planned_tasks() + rect28_tasks():
            print(f"{task.cell:<72} {task.script}.py {task.flags}")
        return 0
    return _common.write_or_check(submits(), ROOT, args=args, generator="imaging_p2a.py")


if __name__ == "__main__":
    sys.exit(main())
