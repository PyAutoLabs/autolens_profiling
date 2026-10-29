"""Likelihood-lifecycle phase 2d: the multi_galaxy + group campaign's task manifest and submits.

autolens_profiling#473 measures the 48 planned ``multi_galaxy`` and ``group`` cells
(``{multi_galaxy,group}/{mge, rectangular, delaunay}``; ``dashboard/coverage.json``) at the
standard size on the pinned release 2026.10.25.1, from a private RAL checkout, with the new cell
scripts under ``scripts/{multi_galaxy,group}/`` (one shared driver,
``scripts/misc/likelihood_breakdown/galaxy_group_standard.py``). This module is the one copy of
that work list: every cell becomes one :class:`Task`, and the submits under ``hpc/`` are rendered
from the tasks with the shared machinery in ``_common.py`` (as phases 2a and 2b)::

    python scripts/misc/campaigns/multi_galaxy_group_p2d.py --write   # (re)render the submits
    python scripts/misc/campaigns/multi_galaxy_group_p2d.py --check   # exit 1 if a submit is stale
    python scripts/misc/campaigns/multi_galaxy_group_p2d.py --list    # print the manifest

DRAFT FOR REVIEW. The cell design (decisions D1-D11 on
``wiki/campaigns/multi_galaxy/likelihood_lifecycle_p2d.md``) is awaiting the human (D12, the
XLA:CPU flag, was decided 2026-10-29), and no submit here has been run. The submits:

- batch 0, the timing arm (decision D9): ``hpc/batch_cpu/submit_campaign_p2d_batch0_ral_cpu_fp64``
  (``--partition=ral`` array of 10) and ``hpc/batch_gpu/submit_campaign_p2d_batch0_a100_fp64``
  (``--gres=gpu:1`` array of 14, ``%2``): every group leg (group bounds multi_galaxy: 1.6x the
  masked pixels, three lens-plane galaxies with mass), writing into the gitignored
  ``output/p2d_batch0/``. Walls are declared guesses (``source: unmeasured``, 3x floor);
- batch 1, one ``--partition=ral`` array and one ``--gres=gpu:1`` job per dataset type
  (``submit_campaign_p2d_<dataset_type>_{ral_cpu,a100}_<instrument>_fp64``), refusing to run
  while :data:`WALL_BASIS_STATUS` is ``provisional`` (batch 0 sets their ``--time`` / ``--mem``).

Stdlib only; nothing here imports the modelling stack except :func:`prepare_dataset`, which the
submits call to simulate a missing dataset under a lock.
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
if str(ROOT / "scripts" / "misc") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "misc"))

import _common  # noqa: E402
from cell_builders import galaxy_group as gg  # noqa: E402  (stdlib at import)

#: The release every cell is measured on (as 2b), and the private RAL checkout at that tag.
PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
#: The shared virtualenv (third-party packages only).
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#473"

DATASET_TYPES = gg.DATASET_TYPES
#: One instrument per dataset type (decision D2): the workspace simulator's pixel scale.
INSTRUMENTS = dict(gg.INSTRUMENTS)
#: The planned variants (decision D1): MGE, rectangular, Delaunay; no delaunay_nn, no sersic.
VARIANTS = gg.VARIANTS
PIXELIZED = gg.PIXELIZED
#: §3.3 "multi_galaxy/*, group/* — as imaging rules" (``catalogue/provisional_designations.json``
#: tables ``parametric`` and ``pixelized``), optimal and secondary per host class.
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
#: ``--config-name`` per (host class, device), as phase 2a.
CONFIG_NAMES = {
    ("hpc_ral_cpu", "cpu"): "hpc_ral_cpu_fp64",
    ("hpc_a100", "gpu"): "hpc_a100_fp64",
    ("hpc_a100", "cpu"): "hpc_a100_cpu_fp64",
}

#: Batch 0 measures the dataset type that bounds the other (decision D9): group, 17.7k masked
#: pixels at 0.1"/px against multi_galaxy's 11.3k at 0.05"/px, with two dPIE extra galaxies.
BATCH0_DATASET_TYPE = "group"
#: Batch 0's declared per-leg wall guess (``source: unmeasured``, 3x floor) and its ``--time``.
#: The imaging hst legs (15k masked pixels, the nearest measured size) took 67-357 s on RAL CPU
#: and 59-329 s on the A100 (``imaging_p2a.BATCH0_WALLS`` delaunay_nn/hst); an hour is ~10x that.
BATCH0_WALL_GUESS_S = 3600
BATCH0_TIME = "4:00:00"
#: Provisional: batch 1 renders unmeasured rows, the placeholder ``--time`` below and a submit
#: that refuses to run until batch 0's walls replace them (README "Setting batch 1's --time").
WALL_BASIS_STATUS = "provisional"
BATCH1_PROVISIONAL_TIME = {"ral_cpu": "4:00:00", "a100": "2-00:00:00"}
#: ``--mem`` until batch 0's sacct MaxRSS is in: the imaging hst legs peaked at 10.2 GiB
#: (``imaging_p2a.BATCH0_PEAK_RSS_KIB``) on a larger image; 32 GB is ~3x that.
PROVISIONAL_MEM_GB = 32
#: The rules batch 1 will take from batch 0 (as 2a / 2b): --time = 1.25x the measured wall plus
#: 15 min, rounded up to 30 min; --mem the next 16/32/64 GB step >= 1.5x the batch-0 MaxRSS.
WALL_HEADROOM = 1.25
WALL_ROUND_S = 1800
WALL_MIN_SLACK_S = 900
MEM_HEADROOM = 1.5
#: Batch 0's measured legs, cell -> (walls.tsv WALL, sacct Elapsed, task). Empty until batch 0.
BATCH0_WALLS: dict = {}
BATCH0_LIMITATIONS: dict = {}


@dataclass(frozen=True, order=True)
class Task:
    """One planned cell: ``<dataset_type>/<variant>/<instrument> <setup_id>@<host> [<measurement>]``."""

    dataset_type: str
    variant: str
    setup_id: str
    host_class: str
    measurement: str

    @property
    def instrument(self) -> str:
        return INSTRUMENTS[self.dataset_type]

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
        return (
            f"{self.dataset_type}/{self.variant}/{self.instrument} "
            f"{self.setup_id}@{self.host_class} [{self.measurement}]"
        )

    @property
    def script(self) -> str:
        """The cell script, relative to ``scripts/`` and without ``.py``."""
        suffix = "_numba" if self.backend == "numba" else ""
        return f"{self.dataset_type}/{self.variant}/likelihood_{self.measurement}{suffix}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id ``<dataset>/<script stem>`` (``wall/check_submits.py``: these
        scripts have no legacy route, so the stem is the canonical one)."""
        return f"{self.dataset_type}/{Path(self.script).name}"

    @property
    def config_name(self) -> str:
        return CONFIG_NAMES[(self.host_class, self.device)]

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (the submits add those)."""
        flags = []
        if self.backend == "jax" and self.operator == "sparse":
            flags.append("--sparse")
        flags += ["--config-name", self.config_name]
        return " ".join(flags)

    @property
    def output(self) -> str:
        """The row's mirrored-tree path, relative to the results root."""
        return (
            f"results/{self.dataset_type}/{self.variant}/{self.instrument}/{self.setup_id}/"
            f"{self.host_class}/{self.measurement}.json"
        )

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS`` for the leg (a numba cell never touches a JAX device)."""
        return "cuda,cpu" if self.device == "gpu" else "cpu"


def planned_tasks() -> list[Task]:
    """Every planned cell of the campaign, sorted (48: 20 RAL CPU + 28 A100)."""
    tasks = []
    for dataset_type in DATASET_TYPES:
        for variant in VARIANTS:
            table = SETUPS["pixelized" if variant in PIXELIZED else "parametric"]
            for host_class, setups in table.items():
                for setup_id in setups:
                    for measurement in MEASUREMENTS:
                        tasks.append(Task(dataset_type, variant, setup_id, host_class, measurement))
    return sorted(tasks)


def batch1_ral_cpu(dataset_type: str) -> list[Task]:
    return [
        t
        for t in planned_tasks()
        if t.host_class == "hpc_ral_cpu" and t.dataset_type == dataset_type
    ]


def batch1_a100(dataset_type: str) -> list[Task]:
    return [
        t for t in planned_tasks() if t.host_class == "hpc_a100" and t.dataset_type == dataset_type
    ]


def batch0_ral_cpu() -> list[Task]:
    return batch1_ral_cpu(BATCH0_DATASET_TYPE)


def batch0_a100() -> list[Task]:
    return batch1_a100(BATCH0_DATASET_TYPE)


def basis_leg(task: Task) -> Task:
    """The batch-0 leg that will bound ``task``: the same variant, setup, host class and
    measurement on the bounding dataset type (decision D9)."""
    return Task(BATCH0_DATASET_TYPE, task.variant, task.setup_id, task.host_class, task.measurement)


def measured_wall(task: Task) -> tuple[int, str]:
    """(seconds, batch-0 task) for ``task`` once batch 0 is in: the larger of WALL and Elapsed."""
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


# --------------------------------------------------------------------------- dataset preflight


def dataset_type_of(instrument: str) -> str:
    """The dataset type planned at ``instrument`` (one each, decision D2)."""
    [dataset_type] = [dt for dt, inst in INSTRUMENTS.items() if inst == instrument]
    return dataset_type


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """Simulate ``dataset/<dataset_type>/<instrument>`` if missing, under a lock (the submits'
    ``prepare_dataset`` hook; array tasks of one dataset start together)."""
    gg.ensure_dataset(dataset_type_of(instrument), root)


# --------------------------------------------------------------------------- submit rendering


def _campaign() -> _common.Campaign:
    return _common.Campaign(
        pin_var="P2D_PIN",
        pin=PIN,
        hpc_base=HPC_BASE,
        private_base=PRIVATE_BASE,
        logs="p2d_logs",
        prefix="p2d",
        module="multi_galaxy_group_p2d",
        allow_var="P2D_ALLOW_PROVISIONAL_WALL",
    )


def render(name: str, tasks: list[Task], *, kind: str, batch: int, title: str) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) / ``a100_job`` / ``a100_array``."""
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    sequential = kind == "a100_job"
    measured = batch == 1 and WALL_BASIS_STATUS == "measured"
    if measured:
        time_basis_s = _common.batch_wall_s(tasks, WALL_BASIS, sequential=sequential)
        time = slurm_time(time_basis_s)
    else:
        time = BATCH0_TIME if batch == 0 else BATCH1_PROVISIONAL_TIME[device]
    n = len(tasks)
    results_root = f"$AP_ROOT/output/p2d_batch{batch}"
    dataset_types = sorted({t.dataset_type for t in tasks})
    sizes = ", ".join(
        f"{dt}/{INSTRUMENTS[dt]} {gg.SPECS[dt].masked_pixels} masked pixels" for dt in dataset_types
    )
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        f"# Likelihood-lifecycle phase 2d multi_galaxy + group campaign ({ISSUE}), batch {batch}: "
        f"{n} cell(s).",
        "# DRAFT FOR REVIEW: the cell design (decisions D1-D11 open, D12 decided; wiki/campaigns/",
        "# multi_galaxy/likelihood_lifecycle_p2d.md) awaits the human; do not submit without a",
        "# batch go.",
        "# GENERATED by scripts/misc/campaigns/multi_galaxy_group_p2d.py --write; edit the generator,",
        "# not this file (the campaign test fails on a hand edit). The batch procedure is in",
        "# scripts/misc/campaigns/README.md.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at tag {PIN} ($PYAUTO_PRIVATE_BASE,",
        f"# default {PRIVATE_BASE}) is prepended to PYTHONPATH over the shared venv",
        f"# ($PYAUTO_HPC_BASE, default {HPC_BASE}, third-party packages only). The job refuses to",
        "# run if a library imports from anywhere else or sits at another tag, and after each leg it",
        "# checks the row's library_revisions equal the checkout's HEADs. The shared mirror is never",
        "# pulled or used. AP_ROOT is the checkout this was submitted from (hpc/<batch>/../..).",
        "#",
        "# CELLS. Every leg runs one cell script of the shared galaxy/group driver on the standard",
        "# cell (MGE source, or Delaunay 1250 + 30 edge / rectangular 35x35) and writes its row",
        "# (runtime legs also parity.json and fit.png) into the mirrored cell named by --output-path",
        f"# under the gitignored output/p2d_batch{batch}/ scratch tree, never in place over results/.",
        "#",
        f"# MEMORY. --mem={PROVISIONAL_MEM_GB}gb, PROVISIONAL ({sizes}): the imaging hst legs",
        "# (15k masked pixels) peaked at 10.2 GiB MaxRSS; batch 0's sacct MaxRSS replaces this",
        f"# (next 16/32/64 GB step >= {MEM_HEADROOM}x the peak) before batch 1.",
    ]
    if kind == "ral_cpu":
        header += [
            "# One cell per array task on `ral` (CPU work never goes on `gpu`). numba cells pin their",
            "# BLAS threads to 1 (production per-worker cost); JAX CPU cells get NPROC=8.",
        ]
    elif kind == "a100_job":
        header += [
            "# One A100 job runs this dataset type's hpc_a100 cells in turn. The jax_cpu_sparse legs",
            "# run on the GPU node's CPU inside this GPU job (JAX_PLATFORMS=cpu, NPROC=8): the",
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
            "# BATCH 0 = TIMING ARM (decision D9). It runs every group leg: group bounds",
            "# multi_galaxy (1.6x the masked pixels, two dPIE extra galaxies). Each leg's wall lands",
            "# in output/p2d_logs/walls.tsv; batch 1's WALL-BASIS rows will cite them",
            "# (source: measured-wall) and its --time / --mem are set from them.",
            "#",
            f"# WALL-BASIS: nothing measured yet; {BATCH0_WALL_GUESS_S} s per leg is a declared guess",
            "# (3x floor), ~10x the imaging hst legs of the nearest measured size.",
        ]
    elif measured:
        header += [
            "# WALL-BASIS: measured by batch 0 on the bounding group leg of the same script, setup",
            f"# and host class; --time is {WALL_HEADROOM}x the submit's total ({time_basis_s}s) plus",
            f"# {WALL_MIN_SLACK_S // 60} min of slack, rounded up to the next {WALL_ROUND_S // 60} min.",
        ]
    else:
        header += [
            "# WALL-BASIS: PROVISIONAL. No cell has a wall yet; batch 0 measures them. Every row is",
            "# `unmeasured` until batch 0 lands, and the job refuses to run while",
            "# WALL_BASIS_STATUS=provisional (export P2D_ALLOW_PROVISIONAL_WALL=1 overrides it).",
        ]
    header += _common.wall_rows(
        tasks,
        device,
        BATCH0_WALL_GUESS_S if batch == 0 else None,
        measured=measured,
        sequential=sequential,
        basis=WALL_BASIS,
    )
    job = f"p2d_b{batch}_{name.split('p2d_', 1)[1]}"
    sbatch = _common.sbatch_block(
        job,
        kind,
        n,
        mem=PROVISIONAL_MEM_GB,
        time=time,
        throttle=("%8" if batch == 0 else "%16") if kind == "ral_cpu" else None,
    )
    campaign = _campaign()
    return _common.assemble(
        header,
        sbatch,
        _common.provisional_guard(WALL_BASIS_STATUS, campaign.allow_var) if batch == 1 else None,
        _common.preamble(campaign, name),
        _common.manifest_arrays(tasks, results_root, module="multi_galaxy_group_p2d"),
        _common.run_block(kind),
    )


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text} for batch 0 and batch 1."""
    out = {}
    name = "submit_campaign_p2d_batch0_ral_cpu_fp64"
    out[f"hpc/batch_cpu/{name}"] = render(
        name,
        batch0_ral_cpu(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm, RAL CPU: every hpc_ral_cpu group cell (the bounding dataset).",
    )
    name = "submit_campaign_p2d_batch0_a100_fp64"
    out[f"hpc/batch_gpu/{name}"] = render(
        name,
        batch0_a100(),
        kind="a100_array",
        batch=0,
        title="Batch-0 timing arm, A100: every hpc_a100 group cell (the bounding dataset).",
    )
    for dataset_type in DATASET_TYPES:
        instrument = INSTRUMENTS[dataset_type]
        name = f"submit_campaign_p2d_{dataset_type}_ral_cpu_{instrument}_fp64"
        out[f"hpc/batch_cpu/{name}"] = render(
            name,
            batch1_ral_cpu(dataset_type),
            kind="ral_cpu",
            batch=1,
            title=f"RAL CPU array: the hpc_ral_cpu {dataset_type} cells at {instrument}.",
        )
        name = f"submit_campaign_p2d_{dataset_type}_a100_{instrument}_fp64"
        out[f"hpc/batch_gpu/{name}"] = render(
            name,
            batch1_a100(dataset_type),
            kind="a100_job",
            batch=1,
            title=f"A100 job: the hpc_a100 {dataset_type} cells at {instrument}.",
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
        for task in planned_tasks():
            print(f"{task.cell:<72} {task.script}.py {task.flags}")
        return 0
    return _common.write_or_check(submits(), ROOT, args=args, generator="multi_galaxy_group_p2d.py")


if __name__ == "__main__":
    sys.exit(main())
