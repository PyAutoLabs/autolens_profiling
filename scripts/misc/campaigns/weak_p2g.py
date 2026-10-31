"""Likelihood-lifecycle phase 2g: the weak-lensing campaign's task manifest and its SLURM submits.

autolens_profiling#476 measures the planned weak-lensing cells (``weak/isothermal/simple``;
``dashboard/coverage.json``) at the standard size (1500 catalogue galaxies, decision D3) on the
pinned release 2026.10.25.1, from a private RAL checkout. Weak lensing had no profiling script and
no planned cell before this campaign; the design (what one evaluation is, the size, the setups and
hosts, the parity reference) is a set of drafted decisions D1-D10 awaiting the human
(``wiki/campaigns/weak/likelihood_lifecycle_p2g.md``). This module is the one copy of the work
list: every runnable cell becomes one :class:`Task`, and the submits under ``hpc/`` are rendered
from the tasks with the machinery in ``_common.py`` (shared with 2a and 2b)::

    python scripts/misc/campaigns/weak_p2g.py --write   # (re)render the submits
    python scripts/misc/campaigns/weak_p2g.py --check   # exit 1 if a submit is stale
    python scripts/misc/campaigns/weak_p2g.py --list    # print the manifest

The submits (README.md beside this file has the batch procedure):

- batch 0, the timing arm and, if the human accepts its rows (decision D9), the whole campaign:
  ``hpc/batch_cpu/submit_campaign_p2g_batch0_ral_cpu_fp64`` (``--partition=ral`` array of 2) and
  ``hpc/batch_gpu/submit_campaign_p2g_batch0_a100_fp64`` (one ``--gres=gpu:1`` job of 4 legs),
  writing into the gitignored ``output/p2g_batch0/``;
- batch 1, the same six legs (``submit_campaign_p2g_weak_ral_cpu_simple_fp64``,
  ``submit_campaign_p2g_weak_a100_simple_fp64``) into ``output/p2g_batch1/``, refusing to run
  until batch 0's walls are in (:data:`WALL_BASIS_STATUS`).

Every planned cell runs: the planner plans a GPU setup only on a GPU host (autolens_profiling#481,
which settles decision D5), so the §3.3 weak row plans 6 cells, not 8, and the CPU-only ``ral``
partition gets no ``jax_gpu_dense_fp64`` cell. Nothing here submits anything.

Stdlib only; nothing here imports the modelling stack except :func:`prepare_dataset`, which the
submits call to simulate a missing catalogue under a lock.
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

#: The release every cell is measured on and the private RAL checkout of the five libraries at
#: that tag (as 2b; ``autolens/weak/`` is identical at the tag and on main, 2026-10-28).
PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#476"

#: The catalogue presets (``scripts/misc/simulators/weak.py``) and the likelihood variant
#: (decisions D2, D3).
INSTRUMENTS = ("simple",)
VARIANT = "isothermal"
#: The §3.3 weak row (the §3.3 copy (retired by #497) table ``weak``):
#: ``jax_cpu_dense_fp64`` (provisional optimal) and ``jax_gpu_dense_fp64`` (secondary), the GPU
#: setup only on the GPU host (``gpu_hosts``, autolens_profiling#481; decision D5).
SETUPS = {
    "hpc_ral_cpu": ("jax_cpu_dense_fp64",),
    "hpc_a100": ("jax_cpu_dense_fp64", "jax_gpu_dense_fp64"),
}
#: Planned (setup_id, host_class) pairs no RAL job can measure. Empty since #481 (decision D5):
#: the planner no longer plans a GPU setup on the CPU-only ``ral`` partition.
UNRUNNABLE: dict[tuple[str, str], str] = {}
MEASUREMENTS = ("runtime", "breakdown")
CONFIG_NAMES = {"hpc_ral_cpu": "hpc_ral_cpu_fp64", "hpc_a100": "hpc_a100_fp64"}

#: Batch 0's declared wall guess per WALL-BASIS row (``source: unmeasured``, 3x floor) and
#: ``--time``. A row covers the legs of one script; the laptop ran each leg in under 50 s
#: (4 threads, dataset simulated in the first), so 1200 s per row is generous: the guess covers a
#: cold A100 JAX initialisation, which the laptop never paid.
BATCH0_WALL_GUESS_S = {"ral_cpu": 1200, "a100": 1200}
BATCH0_TIME = {"ral_cpu": "1:30:00", "a100": "1:30:00"}

#: Batch 0's measured legs: cell -> (walls.tsv WALL, sacct Elapsed, task): RAL array 413293, A100
#: job 413297, jax/jaxlib 0.11.2 (2026-10-29); every leg exit 0, every parity witness ``match``.
#: The A100 job ran its four legs in turn (Elapsed 42 s for all four), so each A100 leg cites its
#: WALL twice. Peak MaxRSS 0.65 GiB (A100 job); the RAL legs ended before sacct sampled one, so
#: the declared --mem stands.
BATCH0_WALLS = {
    "weak/isothermal/simple jax_cpu_dense_fp64@hpc_a100 [breakdown]": (5, 5, "413297"),
    "weak/isothermal/simple jax_cpu_dense_fp64@hpc_a100 [runtime]": (13, 13, "413297"),
    "weak/isothermal/simple jax_cpu_dense_fp64@hpc_ral_cpu [breakdown]": (8, 17, "413293_0"),
    "weak/isothermal/simple jax_cpu_dense_fp64@hpc_ral_cpu [runtime]": (8, 17, "413293_1"),
    "weak/isothermal/simple jax_gpu_dense_fp64@hpc_a100 [breakdown]": (6, 6, "413297"),
    "weak/isothermal/simple jax_gpu_dense_fp64@hpc_a100 [runtime]": (7, 7, "413297"),
}
BATCH0_LIMITATIONS: dict = {}
#: ``provisional`` until batch 0's walls are in: batch 1 renders unmeasured rows and refuses to
#: run (``P2G_ALLOW_PROVISIONAL_WALL=1`` overrides).
WALL_BASIS_STATUS = "measured"
BATCH1_PROVISIONAL_TIME = {"ral_cpu": "1:30:00", "a100": "1:30:00"}
WALL_HEADROOM = 1.25
WALL_ROUND_S = 1800
WALL_MIN_SLACK_S = 900

#: Declared ``--mem`` (GB): the catalogue is 1500 x 2 shears, the JAX compile and a vmap batch of
#: 32 Hessians a few hundred MB (laptop peak RSS ~1 GB). 16 GB on RAL; 32 GB on the A100 node,
#: where the CUDA runtime and the GPU leg's host buffers add to it.
RAL_MEM_GB = 16
A100_MEM_GB = 32


@dataclass(frozen=True, order=True)
class Task:
    """One leg: a planned coverage cell
    ``weak/<variant>/<instrument> <setup_id>@<host_class> [<measurement>]``."""

    variant: str
    instrument: str
    setup_id: str
    host_class: str
    measurement: str
    batch: int = 1

    @property
    def device(self) -> str:
        return self.setup_id.split("_")[1]

    @property
    def cell(self) -> str:
        return (
            f"weak/{self.variant}/{self.instrument} {self.setup_id}@{self.host_class} "
            f"[{self.measurement}]"
        )

    @property
    def script(self) -> str:
        """The leg's script, relative to ``scripts/`` and without ``.py``."""
        return f"weak/{self.variant}/likelihood_{self.measurement}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id. A weak script has no legacy route (it was born in the
        mirrored layout), so ``check_submits`` keys it by its own path: ``weak/<stem>``."""
        return f"weak/{Path(self.script).name}"

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (``run_leg`` adds those)."""
        return f"--config-name {CONFIG_NAMES[self.host_class]}"

    @property
    def output(self) -> str:
        return (
            f"results/weak/{self.variant}/{self.instrument}/{self.setup_id}/{self.host_class}/"
            f"{self.measurement}.json"
        )

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS``: the CPU setup runs on the CPU even on the A100 host (decision D4)."""
        return "cuda,cpu" if self.device == "gpu" else "cpu"


def planned_cells() -> list[Task]:
    """Every planned weak cell (6: the §3.3 reading, the GPU setup on the GPU host only)."""
    return sorted(
        Task(VARIANT, instrument, setup_id, host_class, measurement)
        for instrument in INSTRUMENTS
        for host_class, setups in SETUPS.items()
        for setup_id in setups
        for measurement in MEASUREMENTS
    )


def planned_tasks(batch: int = 1) -> list[Task]:
    """The runnable cells (6: 2 on RAL CPU, 4 on the A100 host)."""
    return [
        replace(t, batch=batch)
        for t in planned_cells()
        if (t.setup_id, t.host_class) not in UNRUNNABLE
    ]


def ral_cpu_tasks(batch: int = 1) -> list[Task]:
    return [t for t in planned_tasks(batch) if t.host_class == "hpc_ral_cpu"]


def a100_tasks(batch: int = 1) -> list[Task]:
    """The A100 job's legs in run order: the CPU setup before the GPU one (the GPU legs then pay
    no first-touch CUDA cost the CPU legs would not)."""
    tasks = [t for t in planned_tasks(batch) if t.host_class == "hpc_a100"]
    return sorted(tasks, key=lambda t: (t.device != "cpu", t.measurement != "runtime"))


def basis_leg(task: Task) -> Task:
    """The batch-0 leg that bounds ``task``: the same cell in batch 0 (one instrument)."""
    return replace(task, batch=0)


def measured_wall(task: Task) -> tuple[int, str]:
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


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """Simulate ``dataset/weak/<instrument>`` if missing, under a lock (array tasks start
    together; without it two would write ``dataset.json`` at once)."""
    import fcntl

    lock_dir = root / "output"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"p2g_dataset_{instrument}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "dataset" / "weak" / instrument / "dataset.json").exists():
            return
        sys.path.insert(0, str(root / "scripts" / "misc"))
        from simulators import weak

        weak.simulate(instrument, root)


# --------------------------------------------------------------------------- submit rendering


def _campaign() -> _common.Campaign:
    return _common.Campaign(
        pin_var="P2G_PIN",
        pin=PIN,
        hpc_base=HPC_BASE,
        private_base=PRIVATE_BASE,
        logs="p2g_logs",
        prefix="p2g",
        module="weak_p2g",
        allow_var="P2G_ALLOW_PROVISIONAL_WALL",
    )


def render(name: str, tasks: list[Task], *, kind: str, batch: int, title: str) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) or ``a100_job``."""
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    sequential = kind == "a100_job"
    measured = batch == 1 and WALL_BASIS_STATUS == "measured"
    if batch == 0:
        guess, time = BATCH0_WALL_GUESS_S[device], BATCH0_TIME[device]
    elif measured:
        guess = None
        time = slurm_time(_common.batch_wall_s(tasks, WALL_BASIS, sequential=sequential))
    else:
        guess, time = None, BATCH1_PROVISIONAL_TIME[device]
    mem = RAL_MEM_GB if device == "ral_cpu" else A100_MEM_GB
    n = len(tasks)
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        f"# Likelihood-lifecycle phase 2g weak-lensing campaign ({ISSUE}), batch {batch}: "
        f"{n} leg(s).",
        "# GENERATED by scripts/misc/campaigns/weak_p2g.py --write; edit the generator,",
        "# not this file (the campaign test fails on a hand edit). The batch procedure is in",
        "# scripts/misc/campaigns/README.md. Decisions adopted by the human 2026-10-29",
        "# (wiki/campaigns/weak/likelihood_lifecycle_p2g.md); a rerun needs the human's batch go.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at tag {PIN} ($PYAUTO_PRIVATE_BASE,",
        f"# default {PRIVATE_BASE}) is prepended to PYTHONPATH over the shared venv",
        f"# ($PYAUTO_HPC_BASE, default {HPC_BASE}, third-party packages only). The job refuses to",
        "# run if a library imports from anywhere else or sits at another tag, and after each leg",
        "# it checks the row's library_revisions equal the checkout's HEADs. The shared mirror is",
        "# never pulled or used. AP_ROOT is the checkout this was submitted from (hpc/<batch>/../..).",
        "#",
        "# CELLS. Every leg evaluates AnalysisWeak.log_likelihood_function on the standard weak",
        "# cell (scripts/misc/likelihood_breakdown/weak_standard.py): the `simple` catalogue of",
        "# 1500 galaxies, a free elliptical isothermal lens at the simulator truth. It writes its",
        "# row, fit.png and parity.json (JIT vs eager NumPy FitWeak) into the mirrored cell named",
        f"# by --output-path under the gitignored output/p2g_batch{batch}/ scratch tree, never in",
        "# place over committed rows.",
        "#",
        f"# MEMORY. --mem={mem}gb, a declared estimate (laptop peak RSS ~1 GB per leg).",
    ]
    if kind == "ral_cpu":
        header += [
            "# One leg per array task on `ral` (CPU work never goes on `gpu`); JAX CPU legs get",
            "# NPROC=8.",
        ]
    else:
        header += [
            "# One A100 job runs the hpc_a100 cells in turn: jax_cpu_dense_fp64 first with",
            "# JAX_PLATFORMS=cpu (the CPU setup on the GPU-present host, decision D4), then",
            "# jax_gpu_dense_fp64 with JAX_PLATFORMS=cuda,cpu.",
        ]
    header += ["#"]
    if batch == 0:
        header += [
            "# BATCH 0 = TIMING ARM. It runs every runnable weak cell once (one instrument, so",
            "# every leg bounds itself). Each leg's wall lands in output/p2g_logs/walls.tsv; batch",
            "# 1's WALL-BASIS rows cite them. If the human accepts batch 0's rows as the campaign's",
            "# (decision D9), batch 1 is not needed.",
            "#",
            f"# WALL-BASIS: nothing measured on RAL yet; {guess} s per row (the legs of one",
            "# script) is a declared guess (3x floor).",
        ]
    elif measured:
        header += [
            "# WALL-BASIS: measured by batch 0. Each leg's wall is that of its batch-0 leg, the",
            "# larger of the walls.tsv WALL and sacct Elapsed; --time is "
            f"{WALL_HEADROOM}x the total plus",
            f"# {WALL_MIN_SLACK_S // 60} min of slack, rounded up to the next {WALL_ROUND_S // 60} min.",
        ]
    else:
        header += [
            "# WALL-BASIS: PROVISIONAL. No leg has a RAL wall yet; batch 0 measures them. Every",
            "# row is `unmeasured` until batch 0 lands, and the job refuses to run while",
            "# WALL_BASIS_STATUS=provisional (export P2G_ALLOW_PROVISIONAL_WALL=1 overrides it).",
        ]
    header += _common.wall_rows(
        tasks, device, guess, measured=measured, sequential=sequential, basis=WALL_BASIS
    )
    job = "p2g_" + name.split("submit_campaign_p2g_", 1)[1].removesuffix("_fp64")
    sbatch = _common.sbatch_block(
        job, kind, n, mem=mem, time=time, throttle="%8" if kind == "ral_cpu" else None
    )
    campaign = _campaign()
    guard = _common.provisional_guard(WALL_BASIS_STATUS, campaign.allow_var) if batch == 1 else None
    return _common.assemble(
        header,
        sbatch,
        guard,
        _common.preamble(campaign, name),
        _common.manifest_arrays(tasks, f"$AP_ROOT/output/p2g_batch{batch}", module="weak_p2g"),
        _common.run_block(kind),
    )


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text}."""
    out = {}

    def add(folder, name, tasks, **kwargs):
        out[f"hpc/{folder}/{name}"] = render(name, tasks, **kwargs)

    add(
        "batch_cpu",
        "submit_campaign_p2g_batch0_ral_cpu_fp64",
        ral_cpu_tasks(0),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm, RAL CPU: the hpc_ral_cpu weak cells.",
    )
    add(
        "batch_gpu",
        "submit_campaign_p2g_batch0_a100_fp64",
        a100_tasks(0),
        kind="a100_job",
        batch=0,
        title="Batch-0 timing arm, A100: the hpc_a100 weak cells.",
    )
    for instrument in INSTRUMENTS:
        add(
            "batch_cpu",
            f"submit_campaign_p2g_weak_ral_cpu_{instrument}_fp64",
            [t for t in ral_cpu_tasks() if t.instrument == instrument],
            kind="ral_cpu",
            batch=1,
            title=f"RAL CPU array: the hpc_ral_cpu weak cells of {instrument}.",
        )
        add(
            "batch_gpu",
            f"submit_campaign_p2g_weak_a100_{instrument}_fp64",
            [t for t in a100_tasks() if t.instrument == instrument],
            kind="a100_job",
            batch=1,
            title=f"A100 job: the hpc_a100 weak cells of {instrument}.",
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
        for task in planned_cells():
            note = UNRUNNABLE.get((task.setup_id, task.host_class))
            tail = f"NOT RUN: {note}" if note else f"{task.script}.py {task.flags}"
            print(f"{task.cell:<70} {tail}")
        return 0
    return _common.write_or_check(submits(), ROOT, args=args, generator="weak_p2g.py")


if __name__ == "__main__":
    sys.exit(main())
