"""Likelihood-lifecycle phase 2c: the point-source campaign's task manifest and its SLURM submits.

autolens_profiling#472 measures the 12 planned point-source cells
(``point_source_image/image_plane`` and ``point_source_source/source_plane``, instrument
``simple``; ``dashboard/coverage.json``) on the pinned release 2026.10.25.1 from a private RAL
checkout, the pin and checkout of phase 2b (decision D1). This module is the one copy of that work
list: every measurable cell becomes one :class:`Task` (script, flags, output path), and the
submits under ``hpc/`` are rendered from the tasks with the machinery the 2a and 2b generators
share (``_common.py``)::

    python scripts/misc/campaigns/point_source_p2c.py --write   # (re)render the submits
    python scripts/misc/campaigns/point_source_p2c.py --check   # exit 1 if a submit is stale
    python scripts/misc/campaigns/point_source_p2c.py --list    # print the manifest

Of the 12 planned cells, 10 are measurable. Two, ``point_source_source/source_plane/simple
jax_gpu_na_fp64@hpc_ral_cpu`` [runtime] and [breakdown], ask for a GPU setup on the RAL CPU host
class, which has no GPU: they get no task and stay ``missing`` (decision D2; see
:data:`MIS_DESIGNATED`). The ``jax_cpu_na_fp64@hpc_a100`` source-plane cells run JAX on the A100
node's CPU (``JAX_PLATFORMS=cpu``) inside the A100 job, as 2a's ``jax_cpu_sparse@hpc_a100`` legs did.

There is no timing batch (decision D3): every evaluation is milliseconds, so one batch runs every
cell under a fixed 1:00:00 wall, checked against the laptop walls of the same scripts
(:data:`LAPTOP_WALLS`):

- ``hpc/batch_cpu/submit_campaign_p2c_point_source_ral_cpu_fp64``: the ``hpc_ral_cpu`` legs, a
  ``--partition=ral`` array;
- ``hpc/batch_gpu/submit_campaign_p2c_point_source_a100_fp64``: the ``hpc_a100`` legs, one
  ``--gres=gpu:1`` job running them in turn.

Both write into the gitignored ``output/p2c/`` tree, never ``results/``.

Besides the cells, each submit runs the solved-likelihood runtime script
(``likelihood_runtime_solved.py``) on the same setup, as an ``experiments/solved_runtime/`` row
(never current; :func:`solved_tasks`). The cell runtime scripts time the free-centre likelihoods
(``FitPositionsImagePairAll``, ``FitPositionsSource``), while the breakdowns, the library default
(``AnalysisPoint``'s ``fit_positions_cls``) and the stale source-plane runtime rows these cells
supersede are the solved ones. Running both lets the human choose which one publishes as the cell
without a second batch.

Stdlib only; nothing here imports the modelling stack.
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

#: The release every cell is measured on and its private RAL checkout (decision D1: the pin,
#: checkout and shared venv of phase 2b).
PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#472"

#: The one point-source instrument: the committed ``dataset/point_source/simple`` positions.
INSTRUMENT = "simple"
#: (dataset_type, variant) of each likelihood.
LIKELIHOODS = (
    ("point_source_image", "image_plane"),
    ("point_source_source", "source_plane"),
)
#: The measurable planned setups per host class (``catalogue/provisional_designations.json``).
SETUPS = {
    "point_source_image": {
        "hpc_ral_cpu": ("jax_cpu_na_fp64",),
        "hpc_a100": ("jax_gpu_na_fp64",),
    },
    "point_source_source": {
        "hpc_ral_cpu": ("jax_cpu_na_fp64",),
        # D2: JAX on the A100 node's CPU (JAX_PLATFORMS=cpu), inside the A100 job.
        "hpc_a100": ("jax_cpu_na_fp64", "jax_gpu_na_fp64"),
    },
}
#: Planned cells with no task (decision D2). provisional_designations.json copies the §3.3 row's
#: host-independent ``secondary`` column (``jax_gpu_dense_fp64``) into every host table of
#: ``point_source_source``, so the RAL CPU host class (no GPU) is planned a GPU setup. Left
#: ``missing`` for a human ruling on the designation, never measured on another host.
MIS_DESIGNATED = {
    ("point_source_source", "source_plane", "jax_gpu_na_fp64", "hpc_ral_cpu"): (
        "designation/planner bug awaiting a human ruling: provisional_designations.json plans a "
        "GPU setup on hpc_ral_cpu (RAL `ral`, no GPU)"
    ),
}
MEASUREMENTS = ("runtime", "breakdown")
CONFIG_NAMES = {
    ("hpc_ral_cpu", "cpu"): "hpc_ral_cpu_fp64",
    ("hpc_a100", "gpu"): "hpc_a100_fp64",
    ("hpc_a100", "cpu"): "hpc_a100_cpu_fp64",
}
#: The ``experiments/`` run name of the solved-likelihood runtime legs.
SOLVED_RUN = "solved_runtime"

#: Decision D3: one batch, fixed walls.
TIME = {"ral_cpu": "1:00:00", "a100": "1:00:00"}
#: The submit's legs must fit WALL_HEADROOM x their laptop walls (the slowest leg of the array,
#: the sum of the A100 job's legs) plus the slack inside TIME. 1.25 is the measured-wall floor
#: (2a and 2b batch 1); the laptop walls are themselves an upper bound (below).
WALL_HEADROOM = 1.25
WALL_MIN_SLACK_S = 900
#: Each script's largest process wall on the laptop (WSL2, NPROC=4, JAX CPU, 2026-10-28) over
#: the smoke runs of this branch (local library main, and the 2026.10.25.1 tags run through the
#: submits' manifest; several ran side by side, so they are inflated). The RAL CPU and A100
#: hosts compiled the image-plane breakdown faster than the laptop in 2026-09
#: (prefix lower + compile + first call: laptop 601 s, RAL CPU 104 s, A100 166 s), so these
#: bound both hosts.
LAPTOP_WALLS = {
    "point_source_image/image_plane/likelihood_runtime": 102,
    "point_source_image/image_plane/likelihood_breakdown": 1051,
    "point_source_image/image_plane/likelihood_runtime_solved": 192,
    "point_source_source/source_plane/likelihood_runtime": 31,
    "point_source_source/source_plane/likelihood_breakdown": 267,
    "point_source_source/source_plane/likelihood_runtime_solved": 35,
}
LAPTOP_REF = "laptop-p2c-smoke-2026-10-28"

#: ``--mem`` (GB). The laptop smokes peaked at 1.1 GB resident (the image-plane breakdown);
#: the 2026-09 point-source submits asked 16-32 GB. 16 GB on `ral`; 32 GB for the A100 job, whose
#: process also maps the CUDA libraries.
MEM_GB = {"ral_cpu": 16, "a100": 32}
LAPTOP_PEAK_RSS_GB = 1.1

_ROUTES = {row["path"]: row["legacy"] for row in load_routes(ROOT)["routes"]}


@dataclass(frozen=True, order=True)
class Task:
    """One leg. ``kind``: ``cell`` (a planned coverage cell,
    ``<dataset_type>/<variant>/simple <setup_id>@<host_class> [<measurement>]``) or ``solved``
    (the solved-likelihood runtime of the same setup, an ``experiments/`` row)."""

    dataset_type: str
    variant: str
    setup_id: str
    host_class: str
    measurement: str
    kind: str = "cell"

    @property
    def instrument(self) -> str:
        return INSTRUMENT

    @property
    def device(self) -> str:
        return self.setup_id.split("_")[1]

    @property
    def cell(self) -> str:
        where = f"{self.dataset_type}/{self.variant}/{INSTRUMENT}"
        if self.kind == "solved":
            return f"{where} {self.setup_id}@{self.host_class} [{SOLVED_RUN} experiment]"
        return f"{where} {self.setup_id}@{self.host_class} [{self.measurement}]"

    @property
    def script(self) -> str:
        """The leg's script, relative to ``scripts/`` and without ``.py``."""
        suffix = "_solved" if self.kind == "solved" else ""
        return f"{self.dataset_type}/{self.variant}/likelihood_{self.measurement}{suffix}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id: ``<dataset_type>/<legacy script stem>``."""
        legacy = _ROUTES[f"scripts/{self.script}.py"]
        return f"{self.dataset_type}/{Path(legacy).stem}"

    @property
    def config_name(self) -> str:
        return CONFIG_NAMES[(self.host_class, self.device)]

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (``run_leg`` adds those; the
        point-source scripts accept ``--instrument`` and always load ``simple``)."""
        return f"--config-name {self.config_name}"

    @property
    def output(self) -> str:
        """The leg's row, relative to the results root."""
        cell = (
            f"results/{self.dataset_type}/{self.variant}/{INSTRUMENT}/{self.setup_id}/"
            f"{self.host_class}"
        )
        if self.kind == "solved":
            return f"{cell}/experiments/{SOLVED_RUN}/runtime.json"
        return f"{cell}/{self.measurement}.json"

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS``: ``cuda`` alone on a GPU leg, so a node without a visible GPU fails
        the leg instead of writing a CPU row under a GPU setup; ``cpu`` otherwise."""
        return "cuda" if self.device == "gpu" else "cpu"


def planned_tasks() -> list[Task]:
    """The 10 measurable planned point-source cells, sorted (4 RAL CPU + 6 A100)."""
    return sorted(
        Task(dataset_type, variant, setup_id, host_class, measurement)
        for dataset_type, variant in LIKELIHOODS
        for host_class, setups in SETUPS[dataset_type].items()
        for setup_id in setups
        for measurement in MEASUREMENTS
    )


def mis_designated_cells() -> list[str]:
    """The planned cells that get no task (decision D2)."""
    return sorted(
        f"{dt}/{variant}/{INSTRUMENT} {setup_id}@{host_class} [{m}]"
        for (dt, variant, setup_id, host_class) in MIS_DESIGNATED
        for m in MEASUREMENTS
    )


def solved_tasks() -> list[Task]:
    """The solved-likelihood runtime of every measurable setup (``experiments/`` rows)."""
    return sorted(
        Task(t.dataset_type, t.variant, t.setup_id, t.host_class, "runtime", "solved")
        for t in planned_tasks()
        if t.measurement == "runtime"
    )


def ral_cpu_tasks() -> list[Task]:
    return [t for t in planned_tasks() + solved_tasks() if t.host_class == "hpc_ral_cpu"]


def a100_tasks() -> list[Task]:
    return [t for t in planned_tasks() + solved_tasks() if t.host_class == "hpc_a100"]


def measured_wall(task: Task) -> tuple[int, str]:
    """(seconds, reference) of ``task``: its script's laptop wall."""
    return LAPTOP_WALLS[task.script], LAPTOP_REF


WALL_BASIS = _common.WallBasis(
    measured_wall=measured_wall,
    basis_leg=lambda task: task,
    limitations={},
    walls={},
    headroom=WALL_HEADROOM,
)


def needed_s(tasks: list[Task], *, sequential: bool) -> float:
    """What ``--time`` must hold: WALL_HEADROOM x (slowest leg, or the sum) plus the slack."""
    return (
        WALL_HEADROOM * _common.batch_wall_s(tasks, WALL_BASIS, sequential=sequential)
        + WALL_MIN_SLACK_S
    )


# --------------------------------------------------------------------------- dataset preflight


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """The submits' per-leg preflight: the point-source dataset is committed
    (``dataset/point_source/simple``), so there is nothing to simulate; refuse if it is missing
    rather than let several legs simulate it at once."""
    dataset = root / "dataset" / "point_source" / instrument / "point_dataset_positions_only.json"
    if not dataset.is_file():
        raise SystemExit(f"missing committed point-source dataset: {dataset}")


# --------------------------------------------------------------------------- submit rendering


def _campaign() -> _common.Campaign:
    return _common.Campaign(
        pin_var="P2C_PIN",
        pin=PIN,
        hpc_base=HPC_BASE,
        private_base=PRIVATE_BASE,
        logs="p2c_logs",
        prefix="p2c",
        module="point_source_p2c",
        allow_var="P2C_ALLOW_PROVISIONAL_WALL",
    )


def render(name: str, tasks: list[Task], *, kind: str, title: str) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) or ``a100_job``."""
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    sequential = kind == "a100_job"
    mem = MEM_GB[device]
    time = TIME[device]
    n = len(tasks)
    total = _common.batch_wall_s(tasks, WALL_BASIS, sequential=sequential)
    n_cells = sum(t.kind == "cell" for t in tasks)
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        f"# Likelihood-lifecycle phase 2c point-source campaign ({ISSUE}): {n} leg(s), "
        f"{n_cells} cell(s).",
        "# GENERATED by scripts/misc/campaigns/point_source_p2c.py --write; edit the generator, not",
        "# this file (the campaign test fails on a hand edit). The procedure is in",
        "# scripts/misc/campaigns/README.md.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at tag {PIN} ($PYAUTO_PRIVATE_BASE,",
        f"# default {PRIVATE_BASE}) is prepended to PYTHONPATH over the shared venv",
        f"# ($PYAUTO_HPC_BASE, default {HPC_BASE}, third-party packages only). The job refuses to",
        "# run if a library imports from anywhere else or sits at another tag, and after each leg it",
        "# checks the row's library_revisions equal the checkout's HEADs. The shared mirror is never",
        "# pulled or used. AP_ROOT is the checkout this was submitted from (hpc/<batch>/../..).",
        "#",
        "# CELLS. Every cell leg runs its likelihood_runtime.py or likelihood_breakdown.py on the",
        "# committed dataset/point_source/simple positions and writes its row, fit.png and",
        "# parity.json (JIT vs the eager NumPy AnalysisPoint(use_jax=False) value) into the",
        "# mirrored cell named by --output-path under the gitignored output/p2c/ scratch tree, never",
        "# in place over committed rows. The solved_runtime legs run likelihood_runtime_solved.py",
        f"# on the same setup into the cell's experiments/{SOLVED_RUN}/ folder (never current).",
        "#",
        f"# MEMORY. --mem={mem}gb: the laptop smokes peaked at {LAPTOP_PEAK_RSS_GB} GB resident; the",
        "# 2026-09 point-source submits asked 16-32 GB.",
    ]
    if kind == "ral_cpu":
        header += [
            "# One leg per array task on `ral` (CPU work never goes on `gpu`); JAX CPU legs get",
            "# NPROC=8.",
        ]
    else:
        header += [
            "# One A100 job runs the hpc_a100 legs in turn. The jax_cpu_na_fp64 source-plane legs",
            "# run on the GPU node's CPU inside this GPU job (JAX_PLATFORMS=cpu, NPROC=8): the",
            "# hpc_a100 host class's CPU setup (decision D2), never a CPU array on the gpu partition.",
            "# The jax_gpu legs set JAX_PLATFORMS=cuda alone, so a leg without a GPU fails instead",
            "# of writing a CPU row.",
        ]
    rule = (
        "the sum of its legs' walls (they run in turn)"
        if sequential
        else "its slowest leg's wall (one leg per array task)"
    )
    header += [
        "#",
        "# WALL-BASIS: no timing batch (decision D3: every evaluation is milliseconds). Each leg's",
        "# wall is its script's largest process wall on the laptop (WSL2, NPROC=4, 2026-10-28, the",
        "# smoke runs of this branch), which compiled the image-plane breakdown 3-6x slower than",
        f"# both RAL hosts in 2026-09. The submit needs {rule}: {total} s;",
        f"# --time={time} holds {WALL_HEADROOM}x that plus {WALL_MIN_SLACK_S // 60} min of slack.",
    ]
    header += _common.wall_rows(
        tasks, device, None, measured=True, sequential=sequential, basis=WALL_BASIS
    )
    job = "p2c_" + name.split("submit_campaign_p2c_", 1)[1].removesuffix("_fp64")
    sbatch = _common.sbatch_block(
        job, kind, n, mem=mem, time=time, throttle="%16" if kind == "ral_cpu" else None
    )
    return _common.assemble(
        header,
        sbatch,
        None,
        _common.preamble(_campaign(), name),
        _common.manifest_arrays(tasks, "$AP_ROOT/output/p2c", module="point_source_p2c"),
        _common.run_block(kind),
    )


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text}."""
    cpu = "submit_campaign_p2c_point_source_ral_cpu_fp64"
    gpu = "submit_campaign_p2c_point_source_a100_fp64"
    return {
        f"hpc/batch_cpu/{cpu}": render(
            cpu,
            ral_cpu_tasks(),
            kind="ral_cpu",
            title="RAL CPU array: the hpc_ral_cpu point-source cells (image-plane and source-plane).",
        ),
        f"hpc/batch_gpu/{gpu}": render(
            gpu,
            a100_tasks(),
            kind="a100_job",
            title="A100 job: the hpc_a100 point-source cells (image-plane and source-plane).",
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="(re)render the submits")
    mode.add_argument("--check", action="store_true", help="exit 1 if a submit is stale")
    mode.add_argument("--list", action="store_true", help="print the manifest")
    args = parser.parse_args(argv)
    if args.list:
        for task in planned_tasks() + solved_tasks():
            print(f"{task.cell:<84} {task.script}.py {task.flags}")
        for cell in mis_designated_cells():
            print(f"{cell:<84} NO TASK (decision D2: mis-designated, awaiting a human ruling)")
        return 0
    return _common.write_or_check(submits(), ROOT, args=args, generator="point_source_p2c.py")


if __name__ == "__main__":
    sys.exit(main())
