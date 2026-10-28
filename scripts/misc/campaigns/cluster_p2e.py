"""Likelihood-lifecycle phase 2e: the cluster campaign's task manifest and its SLURM submits.

autolens_profiling#474 measures the planned cluster cells (``cluster/image_plane`` from the
registry; ``cluster/source_plane`` drafted by analogy with ``point_source_source/source_plane``,
decision D2 on the campaign page) on the pinned release 2026.10.25.1, from a private RAL checkout,
the same pin and checkout as phase 2b. This module is the one copy of that work list: every cell
becomes one :class:`Task` (script, flags, output path), and the submits under ``hpc/`` are
rendered from the tasks with the machinery in ``_common.py`` (shared with 2a and 2b)::

    python scripts/misc/campaigns/cluster_p2e.py --write   # (re)render the submits
    python scripts/misc/campaigns/cluster_p2e.py --check   # exit 1 if a submit is stale
    python scripts/misc/campaigns/cluster_p2e.py --list    # print the manifest

DRAFT FOR REVIEW: nothing here has run on RAL, and the human gives a go per batch. The proposed
batch plan (README.md beside this file):

- batch 0, the timing arm: the image-plane legs, which bound every cluster leg (the source-plane
  likelihood evaluates the same model with no PointSolver): ``--partition=ral`` array of 2
  (``hpc/batch_cpu/submit_campaign_p2e_batch0_ral_cpu_fp64``) and a ``--gres=gpu:1`` array of 2
  (``hpc/batch_gpu/submit_campaign_p2e_batch0_a100_fp64``), with declared wall guesses, into the
  gitignored ``output/p2e_batch0/``;
- batch 1, every runnable cell: one ``--partition=ral`` array
  (``submit_campaign_p2e_cluster_ral_cpu_fp64``) and one ``--gres=gpu:1`` job
  (``submit_campaign_p2e_cluster_a100_fp64``, its legs in turn, the ``jax_cpu`` legs on the A100
  node's CPU), refusing to run while :data:`WALL_BASIS_STATUS` is ``provisional``.

The coverage plan's ``cluster/source_plane jax_gpu_na_fp64@hpc_ral_cpu`` cells (the
§3.3 table's secondary GPU setup carried onto the CPU host) cannot run on a CPU-only partition;
they are listed in :data:`UNRUNNABLE` and left to the human (decision D4), never submitted.

Stdlib only; nothing here imports the modelling stack except :func:`prepare_dataset`, which the
submits call to simulate a missing ``dataset/cluster/simple`` under a lock.
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

#: The release every cell is measured on and the private RAL checkout of the five libraries at
#: that tag (``PYAUTO_PRIVATE_BASE``): the same as phase 2b (#469 ruling R5).
PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
#: The shared virtualenv (third-party packages only).
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#474"

#: Point sources and clusters are instrument-independent; their rows sit under ``simple``.
INSTRUMENT = "simple"
VARIANTS = ("image_plane", "source_plane")
#: The planned (setup_id, host_class) cells per variant: ``dashboard/coverage.json`` (the §3.3
#: table through ``catalogue/provisional_designations.json``). image_plane reads the
#: point_source_image table; source_plane the point_source_source table (decision D2).
SETUPS = {
    "image_plane": (
        ("jax_cpu_na_fp64", "hpc_ral_cpu"),
        ("jax_gpu_na_fp64", "hpc_a100"),
    ),
    "source_plane": (
        ("jax_cpu_na_fp64", "hpc_ral_cpu"),
        ("jax_cpu_na_fp64", "hpc_a100"),
        ("jax_gpu_na_fp64", "hpc_a100"),
        ("jax_gpu_na_fp64", "hpc_ral_cpu"),
    ),
}
#: Planned cells no RAL job can run: a GPU setup on the CPU-only ``ral`` partition (decision D4).
UNRUNNABLE = {
    ("source_plane", "jax_gpu_na_fp64", "hpc_ral_cpu"): (
        "a jax_gpu setup on hpc_ral_cpu: the CPU-only `ral` partition has no GPU, and CPU work "
        "never goes on `gpu`. The §3.3 point_source_source table lists jax_gpu as the secondary "
        "on every host; the human decides whether to scope it out (decision D4)"
    ),
}
MEASUREMENTS = ("runtime", "breakdown")
CONFIG_NAMES = {"hpc_ral_cpu": "hpc_ral_cpu_fp64", "hpc_a100": "hpc_a100_fp64"}

#: The variant whose batch-0 legs bound every leg of the same setup and measurement: the
#: image-plane likelihood is the source-plane model plus a PointSolver forward solve per system.
BOUNDING_VARIANT = "image_plane"
#: Batch 0's declared per-leg wall guesses (``source: unmeasured``, 3x floor) and ``--time``.
#: The basis: the laptop smoke of this branch (WSL2, 8 threads, 2026-10-28) spent ~380 s in
#: lower + compile of each image-plane pipeline with the standard solver (100x100 @ 1.0",
#: precision 0.001); the runtime leg compiles three (plain, solved, vmap) and the breakdown four
#: (two per-system solves, fused plain and solved), so ~20-25 min per leg on the laptop. The
#: historic RAL compiles ran ~2x faster than the laptop's (job 350580). The campaign page has
#: the smoke table. --time holds 3x (guess + the 50 s headline-median allowance).
BATCH0_WALL_GUESS_S = {"ral_cpu": 2400, "a100": 2400}
BATCH0_TIME = {"ral_cpu": "2:30:00", "a100": "2:30:00"}

#: Batch 0's measured legs: cell -> (walls.tsv WALL, sacct Elapsed, array task). Empty until
#: batch 0 runs.
BATCH0_WALLS: dict = {}
#: Batch-0 legs that cannot run, recorded as a known limitation by human ruling.
BATCH0_LIMITATIONS: dict = {}
#: ``measured`` once :data:`BATCH0_WALLS` holds batch 0 and the batch-1 rows cite it; while
#: ``provisional`` batch 1 renders unmeasured rows, :data:`BATCH1_PROVISIONAL_TIME` and a submit
#: that refuses to run (``P2E_ALLOW_PROVISIONAL_WALL=1`` overrides).
WALL_BASIS_STATUS = "provisional"
#: Conservative: RAL one leg per array task (2:30, ~3x the guessed image-plane leg); the A100 job
#: runs its six legs in turn (2 image-plane legs at the guess, 4 source-plane legs of a few
#: minutes each), ~3x their sum.
BATCH1_PROVISIONAL_TIME = {"ral_cpu": "2:30:00", "a100": "5:00:00"}
WALL_HEADROOM = 1.25
WALL_ROUND_S = 1800
WALL_MIN_SLACK_S = 900

#: Declared ``--mem`` (GB): the cluster dataset is a few dozen positions, so memory is JAX's
#: compile temporaries and the vmap batch of the forward solve; the prior point-source and
#: cluster breakdown submits ran in 32 GB. Batch 0's sacct MaxRSS replaces it.
RAL_MEM_GB = 32
A100_MEM_GB = 32
MEM_STEPS_GB = (16, 32, 64, 128, 256)
MEM_HEADROOM = 1.5
BATCH1_MEM_FLOOR_GB = 32
#: Batch 0's peak ``.batch`` MaxRSS (KiB) and the task that reached it. Empty until batch 0 runs.
BATCH0_PEAK_RSS_KIB: dict = {}

_ROUTES = {row["path"]: row["legacy"] for row in load_routes(ROOT)["routes"]}


@dataclass(frozen=True, order=True)
class Task:
    """One leg: the planned coverage cell
    ``cluster/<variant>/simple <setup_id>@<host_class> [<measurement>]``."""

    variant: str
    setup_id: str
    host_class: str
    measurement: str
    batch: int = 1

    instrument = INSTRUMENT

    @property
    def device(self) -> str:
        return self.setup_id.split("_")[1]

    @property
    def cell(self) -> str:
        return (
            f"cluster/{self.variant}/{self.instrument} {self.setup_id}@{self.host_class} "
            f"[{self.measurement}]"
        )

    @property
    def script(self) -> str:
        """The leg's script, relative to ``scripts/`` and without ``.py``."""
        return f"cluster/{self.variant}/likelihood_{self.measurement}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id: ``cluster/<legacy script stem>``."""
        legacy = _ROUTES[f"scripts/{self.script}.py"]
        return f"cluster/{Path(legacy).stem}"

    @property
    def config_name(self) -> str:
        return CONFIG_NAMES[self.host_class]

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (``run_leg`` adds those).

        ``--standard-size`` builds the standard cell: the workspace cluster PointSolver for the
        image plane (decision D1); recorded and inert for the source plane (no solver)."""
        return f"--standard-size --config-name {self.config_name}"

    @property
    def output(self) -> str:
        """The leg's output, relative to the results root."""
        return (
            f"results/cluster/{self.variant}/{self.instrument}/{self.setup_id}/"
            f"{self.host_class}/{self.measurement}.json"
        )

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS``: a ``jax_cpu`` leg runs on the CPU even inside an A100 job."""
        return "cuda,cpu" if self.device == "gpu" else "cpu"


def coverage_cells() -> list[Task]:
    """Every planned cluster cell, runnable or not (12: image_plane 4, source_plane 8)."""
    return sorted(
        Task(variant, setup_id, host_class, measurement)
        for variant, setups in SETUPS.items()
        for setup_id, host_class in setups
        for measurement in MEASUREMENTS
    )


def unrunnable_tasks() -> list[Task]:
    return [t for t in coverage_cells() if (t.variant, t.setup_id, t.host_class) in UNRUNNABLE]


def planned_tasks() -> list[Task]:
    """The runnable cells, sorted (10: RAL CPU 4, A100 6)."""
    return [t for t in coverage_cells() if t not in unrunnable_tasks()]


def batch1_ral_cpu() -> list[Task]:
    return [t for t in planned_tasks() if t.host_class == "hpc_ral_cpu"]


def batch1_a100() -> list[Task]:
    return [t for t in planned_tasks() if t.host_class == "hpc_a100"]


def _bounding(tasks: list[Task]) -> list[Task]:
    from dataclasses import replace

    return [replace(t, batch=0) for t in tasks if t.variant == BOUNDING_VARIANT]


def batch0_ral_cpu() -> list[Task]:
    return _bounding(batch1_ral_cpu())


def batch0_a100() -> list[Task]:
    return _bounding(batch1_a100())


def basis_leg(task: Task) -> Task:
    """The batch-0 leg that bounds ``task``: the image-plane leg of the same measurement on the
    same host class and device (a ``jax_cpu@hpc_a100`` source-plane leg is bounded by the RAL CPU
    image-plane leg of its setup)."""
    from dataclasses import replace

    if task.host_class == "hpc_a100" and task.device == "cpu":
        return replace(task, variant=BOUNDING_VARIANT, host_class="hpc_ral_cpu", batch=0)
    return replace(task, variant=BOUNDING_VARIANT, batch=0)


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


def mem_gb_measured(device: str) -> int:
    """Batch 1's ``--mem``: the next step at or above MEM_HEADROOM x batch 0's peak MaxRSS."""
    peak_gib = BATCH0_PEAK_RSS_KIB[device][0] / 2**20
    return next(
        step for step in MEM_STEPS_GB if step >= max(BATCH1_MEM_FLOOR_GB, MEM_HEADROOM * peak_gib)
    )


# --------------------------------------------------------------------------- dataset preflight


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """Simulate ``dataset/cluster/<instrument>`` if missing, under a lock.

    Array tasks start together; without the lock two would run the cluster simulator into the
    same folder at once and a third could read a half-written ``point_datasets.csv``.
    """
    import fcntl

    if instrument != INSTRUMENT:
        raise ValueError(f"cluster cells use instrument {INSTRUMENT!r}, not {instrument!r}")
    lock_dir = root / "output"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"p2e_dataset_{instrument}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sys.path.insert(0, str(root / "scripts" / "misc"))
        from _profile_cli import auto_simulate_if_missing

        auto_simulate_if_missing(
            root / "dataset" / "cluster" / instrument,
            dataset_type="cluster",
            instrument=instrument,
            workspace_root=root,
        )


# --------------------------------------------------------------------------- submit rendering


def _campaign() -> _common.Campaign:
    return _common.Campaign(
        pin_var="P2E_PIN",
        pin=PIN,
        hpc_base=HPC_BASE,
        private_base=PRIVATE_BASE,
        logs="p2e_logs",
        prefix="p2e",
        module="cluster_p2e",
        allow_var="P2E_ALLOW_PROVISIONAL_WALL",
    )


def _mem(device: str, batch: int) -> tuple[int, str]:
    """(``--mem`` GB, the header line saying where it came from)."""
    if batch == 1 and WALL_BASIS_STATUS == "measured":
        mem = mem_gb_measured(device)
        return mem, (
            f"# MEMORY. --mem={mem}gb from batch 0's sacct MaxRSS x {MEM_HEADROOM}, at least "
            f"{BATCH1_MEM_FLOOR_GB} GB."
        )
    mem = RAL_MEM_GB if device == "ral_cpu" else A100_MEM_GB
    return mem, (
        f"# MEMORY. --mem={mem}gb, a declared estimate (a few dozen positions; JAX compile"
        "\n# temporaries and the vmap batch dominate). Batch 0's sacct MaxRSS replaces it."
    )


def render(name: str, tasks: list[Task], *, kind: str, batch: int, title: str) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) / ``a100_job`` / ``a100_array``."""
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
    mem, mem_line = _mem(device, batch)
    n = len(tasks)
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        f"# Likelihood-lifecycle phase 2e cluster campaign ({ISSUE}), batch {batch}: {n} leg(s).",
        "# GENERATED by scripts/misc/campaigns/cluster_p2e.py --write; edit the generator,",
        "# not this file (the campaign test fails on a hand edit). The batch procedure is in",
        "# scripts/misc/campaigns/README.md. DRAFT: the human gives a go per batch.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at tag {PIN} ($PYAUTO_PRIVATE_BASE,",
        f"# default {PRIVATE_BASE}) is prepended to PYTHONPATH over the shared venv",
        f"# ($PYAUTO_HPC_BASE, default {HPC_BASE}, third-party packages only). The job refuses to",
        "# run if a library imports from anywhere else or sits at another tag, and after each leg",
        "# it checks the row's library_revisions equal the checkout's HEADs. The shared mirror is",
        "# never pulled or used. AP_ROOT is the checkout this was submitted from (hpc/<batch>/../..).",
        "#",
        "# CELLS. Every leg runs the standard cluster cell (--standard-size): dataset/cluster/simple",
        "# (2 main dPIE + 10 scaling dPIE + NFW host; 2 sources at z = 1, 2), the image plane with the",
        '# workspace cluster PointSolver (100x100 @ 1.0", precision 0.001, magnification_threshold',
        "# 0.1, residual_threshold 10, all explicit; decision D1). Each leg writes its row, fit.png and",
        "# (runtime) parity.json into the mirrored cell named by --output-path under the gitignored",
        f"# output/p2e_batch{batch}/ scratch tree, never in place over committed rows.",
        "#",
        mem_line,
    ]
    if kind == "ral_cpu":
        header += ["# One leg per array task on `ral` (CPU work never goes on `gpu`); NPROC=8."]
    elif kind == "a100_job":
        header += [
            "# One A100 job runs the hpc_a100 cells in turn; the jax_cpu legs run on the A100",
            "# node's CPU (JAX_PLATFORMS=cpu), the jax_gpu legs on the GPU.",
        ]
    else:
        header += ["# One leg per array task, each with --gres=gpu:1, throttled to %2."]
    header += ["#"]
    if batch == 0:
        header += [
            "# BATCH 0 = TIMING ARM. It runs the image-plane legs, which bound every cluster leg of",
            "# the same setup and measurement (the source plane is the same model without the",
            "# PointSolver). Each leg's wall lands in output/p2e_logs/walls.tsv; batch 1's WALL-BASIS",
            "# rows cite them and its --time is set from them.",
            "#",
            f"# WALL-BASIS: nothing measured with the standard solver on RAL yet; {guess} s per leg is",
            "# a declared guess (3x floor). The timing arm exists to find out.",
        ]
    elif measured:
        header += [
            "# WALL-BASIS: measured by batch 0. Each leg's wall is that of its bounding image-plane",
            "# batch-0 leg, the larger of the walls.tsv WALL and sacct Elapsed; --time is",
            f"# {WALL_HEADROOM}x the total plus {WALL_MIN_SLACK_S // 60} min of slack, rounded up to",
            f"# the next {WALL_ROUND_S // 60} min.",
        ]
    else:
        header += [
            "# WALL-BASIS: PROVISIONAL. No leg has a standard-cell wall on RAL yet; batch 0 measures",
            "# them. Every row is `unmeasured` until batch 0 lands, and the job refuses to run while",
            "# WALL_BASIS_STATUS=provisional (export P2E_ALLOW_PROVISIONAL_WALL=1 overrides it).",
        ]
    header += _common.wall_rows(
        tasks, device, guess, measured=measured, sequential=sequential, basis=WALL_BASIS
    )
    job = "p2e_" + name.split("submit_campaign_p2e_", 1)[1].removesuffix("_fp64")
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
        _common.manifest_arrays(tasks, f"$AP_ROOT/output/p2e_batch{batch}", module="cluster_p2e"),
        _common.run_block(kind),
    )


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text}."""
    out = {}

    def add(folder, name, tasks, **kwargs):
        out[f"hpc/{folder}/{name}"] = render(name, tasks, **kwargs)

    add(
        "batch_cpu",
        "submit_campaign_p2e_batch0_ral_cpu_fp64",
        batch0_ral_cpu(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm, RAL CPU: the image-plane hpc_ral_cpu cluster legs.",
    )
    add(
        "batch_gpu",
        "submit_campaign_p2e_batch0_a100_fp64",
        batch0_a100(),
        kind="a100_array",
        batch=0,
        title="Batch-0 timing arm, A100: the image-plane hpc_a100 cluster legs.",
    )
    add(
        "batch_cpu",
        "submit_campaign_p2e_cluster_ral_cpu_fp64",
        batch1_ral_cpu(),
        kind="ral_cpu",
        batch=1,
        title="RAL CPU array: every hpc_ral_cpu cluster cell (image plane and source plane).",
    )
    add(
        "batch_gpu",
        "submit_campaign_p2e_cluster_a100_fp64",
        batch1_a100(),
        kind="a100_job",
        batch=1,
        title="A100 job: every hpc_a100 cluster cell (jax_gpu on the GPU, jax_cpu on its CPU).",
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
            print(f"{task.cell:<70} {task.script}.py {task.flags}")
        for task in unrunnable_tasks():
            print(f"{task.cell:<70} NOT SUBMITTED (decision D4)")
        return 0
    return _common.write_or_check(submits(), ROOT, args=args, generator="cluster_p2e.py")


if __name__ == "__main__":
    sys.exit(main())
