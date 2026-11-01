"""Likelihood-lifecycle phase 6: the gradient campaign's task manifest and its SLURM submits.

autolens_profiling#507 (6b). Phase 6 measures, for every Optimal setup, what a gradient
evaluation of the likelihood costs beside the forward call and whether it is correct
(``scripts/misc/gradient/harness.py``, ``scripts/misc/tooling/gradient_witness.py``). The cells
(human ruling 2026-11-01, #505):

- every Pulse designation whose optimal setup is JAX: that setup on its reference host;
- every designation whose optimal setup is numba (which JAX cannot differentiate): its JAX CPU
  secondary on the CPU host, the setup a gradient sampler would run;
- the held group (``catalogue/designation_holds.json``): its provisional JAX optimal setups,
  regime-only routes (``_streaming``) excluded as in planning.

Each gradient leg is the phase-2 runtime leg of the same cell with its script swapped for the
likelihood's ``gradient_breakdown.py`` (which runs the runtime script with ``--gradient``): the
same flags, instrument, platform and private checkout, so the gradient is taken of exactly the
likelihood the day-zero row timed. Every leg runs on the day-zero tag ``2026.10.25.1`` (fp64),
so its record can carry the ``day_zero`` stamp (ruling on #506)::

    python scripts/misc/campaigns/gradient_p6.py --list    # print the manifest
    python scripts/misc/campaigns/gradient_p6.py --write   # (re)render the submits
    python scripts/misc/campaigns/gradient_p6.py --check   # exit 1 if a submit is stale

The submits (``hpc/batch_cpu/submit_campaign_p6_gradient_<source>_ral_cpu_fp64`` and
``hpc/batch_gpu/submit_campaign_p6_gradient_<source>_a100_fp64``, one pair per phase-2 source
campaign, whose ``prepare_dataset`` they reuse) write ``gradient.json`` into the gitignored
``output/p6_gradient/`` scratch tree. Nothing here submits anything; a run needs the human's go,
from a session that can reach RAL.

Walls (6c, autolens_profiling#509): the first run (2026-11-01, witness version 1, RAL 413707-413722)
measured 69 of the 74 legs; ``P6_WALLS`` holds them and their rows are ``measured-wall``. Version 2
adds the FD step sweep (``gradient_witness.SWEEP_STEPS``, ``SWEEP_CALLS`` more forward calls per
leg), so each measured wall gains ``SWEEP_CALLS`` x the cell's measured single-call forward time.
The five imaging RAL CPU legs killed at 64 GB (``P6_OOM``) measured nothing: their rows stay
``unmeasured`` (``probe-first: yes``, the 4x runtime-wall guess, 3x floor) and the imaging RAL CPU
submit asks ``GRADIENT_MEM_GB``.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE, ROOT / "scripts" / "misc" / "tooling"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _common  # noqa: E402
import catalogue_lifecycle as lc  # noqa: E402
import gradient_witness as gw  # noqa: E402

PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#507, #509"
RESULTS = "$AP_ROOT/output/p6_gradient"
#: The phase-2 source campaigns, whose runtime legs the gradient legs follow.
SOURCES = (
    "imaging_p2a",
    "interferometer_p2b",
    "point_source_p2c",
    "multi_galaxy_group_p2d",
    "cluster_p2e",
    "multi_dataset_datacube_p2f",
    "weak_p2g",
)
#: A gradient leg's declared wall guess: GUESS_FACTOR x the cell's measured phase-2 runtime wall
#: (forward and value_and_grad compiles, vmap of both, 6 forward FD calls), at least GUESS_FLOOR_S.
GUESS_FACTOR = 4
GUESS_FLOOR_S = 1200
GUESS_HEADROOM = 3.0
WALL_ROUND_S = 1800
#: A measured row's headroom and slack (as the phase-2 batch-1 rows).
WALL_HEADROOM = 1.25
WALL_MIN_SLACK_S = 900
#: Forward calls the version-2 FD step sweep adds per leg (two per step per direction).
SWEEP_CALLS = 2 * len(gw.SWEEP_STEPS) * gw.FD_DIRECTIONS
#: --mem (GB) over the source campaign's value: the five imaging RAL CPU legs below were killed at
#: 64 GB inside the gradient measurement (their measured siblings peaked at 38-48 GB, sacct
#: MaxRSS); RAL CPU nodes hold 921-976 GB.
GRADIENT_MEM_GB = {("imaging_p2a", "ral_cpu"): 192}

#: The first run's measured legs (2026-11-01, RAL 413707-413722, jax 0.11.2, 2026.10.25.1):
#: cell -> (walls.tsv WALL s, sacct Elapsed s, array task, forward single-call s from the leg's
#: gradient.json). Every leg wrote a valid witness; the eight imaging/mge legs exited 4 only
#: because the 6a pilot wrote outside --output-path (fixed in #509; the witnesses were moved).
P6_WALLS = {
    "cluster/image_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (
        561,
        566,
        "413713_0",
        0.0054,
    ),
    "cluster/image_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (
        734,
        743,
        "413714_0",
        0.002676,
    ),
    "cluster/source_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (
        77,
        85,
        "413713_1",
        0.0002732,
    ),
    "cluster/source_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (
        156,
        168,
        "413714_1",
        0.0005694,
    ),
    "datacube/delaunay/alma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        685,
        691,
        "413721_0",
        13.85,
    ),
    "datacube/delaunay/alma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        282,
        284,
        "413722_0",
        0.6244,
    ),
    "datacube/delaunay/alma_high jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        1089,
        1095,
        "413721_1",
        20.33,
    ),
    "datacube/delaunay/alma_high jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        366,
        368,
        "413722_1",
        0.7342,
    ),
    "datacube/delaunay/sma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        1068,
        1074,
        "413721_2",
        11.22,
    ),
    "datacube/delaunay/sma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        218,
        222,
        "413722_2",
        0.5723,
    ),
    "group/delaunay/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        2109,
        2113,
        "413715_0",
        5.418,
    ),
    "group/delaunay/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (173, 183, "413716_0", 0.05486),
    "group/mge/euclid jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (130, 134, "413715_1", 0.04043),
    "group/mge/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (200, 210, "413716_1", 0.007131),
    "group/rectangular/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        2106,
        2110,
        "413715_2",
        4.859,
    ),
    "group/rectangular/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        158,
        160,
        "413716_2",
        0.04361,
    ),
    "imaging/delaunay/ao jax_gpu_dense_fp64@hpc_a100 [gradient]": (236, 246, "413718_0", 0.3467),
    "imaging/delaunay/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        525,
        529,
        "413717_1",
        1.283,
    ),
    "imaging/delaunay/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        164,
        168,
        "413718_1",
        0.04116,
    ),
    "imaging/delaunay/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        1249,
        1253,
        "413717_2",
        3.076,
    ),
    "imaging/delaunay/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (232, 235, "413718_2", 0.05553),
    "imaging/delaunay/jwst jax_gpu_dense_fp64@hpc_a100 [gradient]": (228, 231, "413718_3", 0.08294),
    "imaging/delaunay_nn/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        258,
        260,
        "413718_4",
        0.08581,
    ),
    "imaging/mge/ao jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (315, 322, "413717_5", 0.4458),
    "imaging/mge/ao jax_gpu_dense_fp64@hpc_a100 [gradient]": (142, 144, "413718_5", 0.01533),
    "imaging/mge/euclid jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (59, 65, "413717_6", 0.008209),
    "imaging/mge/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (104, 107, "413718_6", 0.004991),
    "imaging/mge/hst jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (73, 79, "413717_7", 0.02054),
    "imaging/mge/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (115, 118, "413718_7", 0.00567),
    "imaging/mge/jwst jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (80, 87, "413717_8", 0.0478),
    "imaging/mge/jwst jax_gpu_dense_fp64@hpc_a100 [gradient]": (107, 109, "413718_8", 0.006279),
    "imaging/rectangular/ao jax_gpu_dense_fp64@hpc_a100 [gradient]": (200, 202, "413718_9", 0.2966),
    "imaging/rectangular/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        466,
        468,
        "413717_10",
        0.9634,
    ),
    "imaging/rectangular/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        144,
        147,
        "413718_10",
        0.0283,
    ),
    "imaging/rectangular/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        1225,
        1228,
        "413717_11",
        2.725,
    ),
    "imaging/rectangular/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        215,
        217,
        "413718_11",
        0.03851,
    ),
    "imaging/rectangular/jwst jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        217,
        219,
        "413718_12",
        0.06334,
    ),
    "interferometer/delaunay/alma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        158,
        164,
        "413719_0",
        2.283,
    ),
    "interferometer/delaunay/alma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        63,
        66,
        "413720_0",
        0.03826,
    ),
    "interferometer/delaunay/alma_high jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        486,
        492,
        "413719_1",
        8.134,
    ),
    "interferometer/delaunay/alma_high jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        92,
        95,
        "413720_1",
        0.07713,
    ),
    "interferometer/delaunay/jvla jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        2871,
        2877,
        "413719_2",
        50.17,
    ),
    "interferometer/delaunay/jvla jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        268,
        271,
        "413720_2",
        0.4358,
    ),
    "interferometer/delaunay/sma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        310,
        316,
        "413719_3",
        0.7976,
    ),
    "interferometer/delaunay/sma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        44,
        46,
        "413720_3",
        0.02694,
    ),
    "interferometer/mge/sma jax_cpu_dense_fp64_nufft@hpc_ral_cpu [gradient]": (
        59,
        65,
        "413719_4",
        0.0827,
    ),
    "interferometer/mge/sma jax_gpu_dense_fp64_nufft@hpc_a100 [gradient]": (
        39,
        42,
        "413720_4",
        0.003552,
    ),
    "interferometer/rectangular/alma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        136,
        142,
        "413719_5",
        2.055,
    ),
    "interferometer/rectangular/alma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        40,
        43,
        "413720_5",
        0.03412,
    ),
    "interferometer/rectangular/alma_high jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        461,
        467,
        "413719_6",
        7.501,
    ),
    "interferometer/rectangular/alma_high jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        72,
        74,
        "413720_6",
        0.06822,
    ),
    "interferometer/rectangular/jvla jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        2923,
        2929,
        "413719_7",
        49.52,
    ),
    "interferometer/rectangular/jvla jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        244,
        246,
        "413720_7",
        0.4346,
    ),
    "interferometer/rectangular/sma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        242,
        249,
        "413719_8",
        0.637,
    ),
    "interferometer/rectangular/sma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        32,
        34,
        "413720_8",
        0.02251,
    ),
    "multi_dataset/delaunay/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        2233,
        2239,
        "413721_3",
        5.942,
    ),
    "multi_dataset/delaunay/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        402,
        406,
        "413722_3",
        0.1086,
    ),
    "multi_galaxy/delaunay/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        1503,
        1507,
        "413715_3",
        3.846,
    ),
    "multi_galaxy/delaunay/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        176,
        179,
        "413716_3",
        0.04898,
    ),
    "multi_galaxy/mge/hst jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (
        116,
        122,
        "413715_4",
        0.02637,
    ),
    "multi_galaxy/mge/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (196, 203, "413716_4", 0.006891),
    "multi_galaxy/rectangular/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        1466,
        1469,
        "413715_5",
        3.444,
    ),
    "multi_galaxy/rectangular/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        158,
        161,
        "413716_5",
        0.03706,
    ),
    "point_source_image/image_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (
        49,
        52,
        "413709_0",
        0.001609,
    ),
    "point_source_image/image_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (
        75,
        86,
        "413711_0",
        0.0008786,
    ),
    "point_source_source/source_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (
        15,
        17,
        "413709_1",
        0.0001714,
    ),
    "point_source_source/source_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (
        31,
        41,
        "413711_1",
        0.0002468,
    ),
    "weak/isothermal/simple jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (
        13,
        24,
        "413707_0",
        0.0001476,
    ),
    "weak/isothermal/simple jax_gpu_dense_fp64@hpc_a100 [gradient]": (
        16,
        26,
        "413708_0",
        0.0002463,
    ),
}
#: Killed at --mem=64gb (OUT_OF_MEMORY) inside the gradient measurement: cell -> (Elapsed s, task).
P6_OOM = {
    "imaging/delaunay/ao jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (1054, "413717_0"),
    "imaging/delaunay/jwst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (1055, "413717_3"),
    "imaging/delaunay_nn/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (1475, "413717_4"),
    "imaging/rectangular/ao jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (907, "413717_9"),
    "imaging/rectangular/jwst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (971, "413717_12"),
}


def _module(name: str):
    return importlib.import_module(name)


def gradient_setups(dataset_type: str, variant: str, host_class: str) -> set[str]:
    """The setups phase 6 differentiates for one likelihood on one reference host."""
    entry = lc.designation_of(dataset_type, variant, host_class)
    if entry is not None:
        optimal = entry["optimal"]
        if optimal.startswith("jax_"):
            return {optimal}
        return {s for s in entry["secondary"] if s.startswith("jax_cpu_")}
    hold = lc.hold_of(dataset_type, variant, host_class)
    if hold is not None:
        return {
            s for s in hold["optimal"] if s.startswith("jax_") and not s.endswith(lc._REGIME_ONLY)
        }
    return set()


def _dataset_type(task) -> str:
    """The phase-2 tasks name their dataset type in different fields (or by campaign)."""
    for field in ("dataset_type", "dataset"):
        value = getattr(task, field, None)
        if value:
            return value
    return task.cell.split("/", 1)[0]


@dataclass(frozen=True, order=True)
class GradientTask:
    """One gradient leg, following the phase-2 runtime leg ``source_task`` of the same cell."""

    source: str
    dataset_type: str
    variant: str
    instrument: str
    setup_id: str
    host_class: str
    runtime_flags: str
    runtime_output: str
    platform: str
    runtime_wall_s: int
    measurement: str = "gradient"

    @property
    def device(self) -> str:
        return self.setup_id.split("_")[1]

    @property
    def cell(self) -> str:
        return (
            f"{self.dataset_type}/{self.variant}/{self.instrument} "
            f"{self.setup_id}@{self.host_class} [gradient]"
        )

    @property
    def script(self) -> str:
        return f"{self.dataset_type}/{self.variant}/gradient_breakdown"

    @property
    def gate_cell(self) -> str:
        # check_submits keys an invocation by (dataset folder, script stem).
        return f"{self.dataset_type}/gradient_breakdown"

    @property
    def flags(self) -> str:
        return self.runtime_flags

    @property
    def output(self) -> str:
        return str(Path(self.runtime_output).with_name("gradient.json"))

    @property
    def guess_s(self) -> int:
        return max(GUESS_FLOOR_S, GUESS_FACTOR * self.runtime_wall_s)


def planned_tasks() -> list[GradientTask]:
    """Every gradient leg, derived from the phase-2 runtime legs."""
    tasks = []
    for source in SOURCES:
        module = _module(source)
        for t in module.planned_tasks():
            if t.measurement != "runtime":
                continue
            dataset_type = _dataset_type(t)
            if t.setup_id not in gradient_setups(dataset_type, t.variant, t.host_class):
                continue
            try:
                wall = int(module.measured_wall(t)[0])
            except (KeyError, AttributeError):
                wall = GUESS_FLOOR_S
            tasks.append(
                GradientTask(
                    source=source,
                    dataset_type=dataset_type,
                    variant=t.variant,
                    instrument=getattr(t, "instrument", None) or "simple",
                    setup_id=t.setup_id,
                    host_class=t.host_class,
                    runtime_flags=t.flags,
                    runtime_output=t.output,
                    platform=t.platform,
                    runtime_wall_s=wall,
                )
            )
    return sorted(set(tasks))


def measured_wall(task: GradientTask) -> tuple[int, str]:
    """The first run's wall for ``task`` plus the version-2 sweep's extra forward calls."""
    wall, elapsed, ref, forward_s = P6_WALLS[task.cell]
    return max(wall, elapsed) + math.ceil(SWEEP_CALLS * forward_s), ref


WALL_BASIS = _common.WallBasis(
    measured_wall=measured_wall,
    basis_leg=lambda task: task,
    limitations={},
    walls={cell: row[:3] for cell, row in P6_WALLS.items()},
    headroom=WALL_HEADROOM,
)


def _mem(source: str, kind: str, tasks) -> int:
    if (source, kind) in GRADIENT_MEM_GB:
        return GRADIENT_MEM_GB[(source, kind)]
    module = _module(source)
    value = getattr(module, "RAL_MEM_GB" if kind == "ral_cpu" else "A100_MEM_GB", 64)
    if isinstance(value, dict):
        return max(value.get(t.instrument, max(value.values())) for t in tasks)
    return int(value)


def _campaign(source: str) -> _common.Campaign:
    return _common.Campaign(
        pin_var="P6_PIN",
        pin=PIN,
        hpc_base=HPC_BASE,
        private_base=PRIVATE_BASE,
        logs="p6_gradient_logs",
        prefix="p6",
        module=source,
        allow_var="P6_ALLOW_PROVISIONAL_WALL",
    )


def render(name: str, source: str, tasks: list[GradientTask], *, kind: str) -> str:
    """One submit: an array of gradient legs (one leg per task)."""
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    measured = [t for t in tasks if t.cell in P6_WALLS]
    unmeasured = [t for t in tasks if t.cell not in P6_WALLS]
    guess = max((t.guess_s for t in unmeasured), default=None)
    budgets = []
    if measured:
        wall = _common.batch_wall_s(measured, WALL_BASIS, sequential=False)
        budgets.append(
            _common.slurm_time(
                wall, headroom=WALL_HEADROOM, slack_s=WALL_MIN_SLACK_S, round_s=WALL_ROUND_S
            )
        )
    if unmeasured:
        budgets.append(
            _common.slurm_time(guess, headroom=GUESS_HEADROOM, slack_s=0, round_s=WALL_ROUND_S)
        )
    time = max(budgets, key=_slurm_seconds)
    mem = _mem(source, kind, tasks)
    source_mem = "the phase-2 source campaign's declared value for these cells"
    if (source, kind) in GRADIENT_MEM_GB:
        source_mem = "raised over the source campaign's value (legs killed at 64 GB in 6c)"
    header = [
        "#!/bin/bash -l",
        "#",
        f"# Likelihood-lifecycle phase 6 gradient campaign ({ISSUE}): {len(tasks)} leg(s) from",
        f"# the {source} runtime cells, {'RAL CPU' if kind == 'ral_cpu' else 'A100'}.",
        "# GENERATED by scripts/misc/campaigns/gradient_p6.py --write; edit the generator,",
        "# not this file (the campaign test fails on a hand edit). A run needs the human's go.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at the day-zero tag {PIN}",
        f"# ($PYAUTO_PRIVATE_BASE, default {PRIVATE_BASE}) is prepended to PYTHONPATH over the",
        f"# shared venv ($PYAUTO_HPC_BASE, default {HPC_BASE}). The job refuses to run if a",
        "# library imports from anywhere else or sits at another tag, and after each leg it checks",
        "# the witness's library_revisions equal the checkout's HEADs.",
        "#",
        "# CELLS. Each leg runs the likelihood's gradient_breakdown.py with the flags of the cell's",
        "# phase-2 runtime leg: the runtime script builds its own fused likelihood and, with",
        "# --gradient, hands it to scripts/misc/gradient/harness.py (forward vs value_and_grad, the",
        "# pre-registered finite-difference check and FD step sweep) instead of timing it. It writes",
        "# gradient.json (witness version 2)",
        "# into the mirrored cell named by --output-path under the gitignored output/p6_gradient/",
        "# scratch tree, never in place over committed rows. fp64 only.",
        "#",
        f"# MEMORY. --mem={mem}gb, {source_mem}.",
        "#",
        "# WALL-BASIS: a measured row cites the first run's wall of its legs (the slowest; array",
        f"# tasks) plus {SWEEP_CALLS} forward calls of the FD step sweep at the cell's measured",
        "# forward time. An unmeasured row is a leg the first run did not finish: a declared guess,",
        f"# {GUESS_FACTOR}x the cell's measured phase-2 runtime wall (at least {GUESS_FLOOR_S} s),",
        "# with the 3x floor. The legs' walls land in output/p6_gradient_logs/walls.tsv.",
    ]
    if measured:
        header += _common.wall_rows(
            measured, device, None, measured=True, sequential=False, basis=WALL_BASIS
        )
    if unmeasured:
        header += _common.wall_rows(
            unmeasured, device, guess, measured=False, sequential=False, basis=None
        )
    job = "p6_" + name.split("submit_campaign_p6_gradient_", 1)[1].removesuffix("_fp64")
    sbatch = _common.sbatch_block(
        job,
        "ral_cpu" if kind == "ral_cpu" else "a100_array",
        len(tasks),
        mem=mem,
        time=time,
        throttle="%8" if kind == "ral_cpu" else None,
    )
    return _common.assemble(
        header,
        sbatch,
        None,
        _common.preamble(_campaign(source), name),
        _common.manifest_arrays(tasks, RESULTS, module="gradient_p6"),
        _common.run_block("ral_cpu" if kind == "ral_cpu" else "a100_array"),
    )


def _slurm_seconds(value: str) -> int:
    h, m, sec = (int(x) for x in value.split(":"))
    return h * 3600 + m * 60 + sec


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text}."""
    out = {}
    tasks = planned_tasks()
    for source in SOURCES:
        for kind, host, folder, tag in (
            ("ral_cpu", "hpc_ral_cpu", "batch_cpu", "ral_cpu"),
            ("a100", "hpc_a100", "batch_gpu", "a100"),
        ):
            legs = [t for t in tasks if t.source == source and t.host_class == host]
            if not legs:
                continue
            short = source.rsplit("_p2", 1)[0]
            name = f"submit_campaign_p6_gradient_{short}_{tag}_fp64"
            out[f"hpc/{folder}/{name}"] = render(name, source, legs, kind=kind)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        tasks = planned_tasks()
        for t in tasks:
            print(f"{t.cell}  <- {t.source} (runtime wall {t.runtime_wall_s} s)")
        print(f"{len(tasks)} gradient legs")
        return 0
    return _common.write_or_check(submits(), ROOT, args=args, generator="gradient_p6.py")


if __name__ == "__main__":
    raise SystemExit(main())
