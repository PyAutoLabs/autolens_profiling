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

Walls (6c, autolens_profiling#509): the version-2 run (2026-11-01, RAL 413791-413830, FD step
sweep included) measured 72 of the 74 legs; ``P6_WALLS`` holds them and their rows are
``measured-wall``. (The version-1 run, RAL 413707-413722, measured 69 without the sweep; it is
superseded.) The two imaging RAL CPU legs killed at 192 GB (``P6_OOM``) measured nothing: their rows
stay ``unmeasured`` (``probe-first: yes``, the 4x runtime-wall guess, 3x floor) and the imaging RAL
CPU submit asks ``GRADIENT_MEM_GB``.
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
#: --mem (GB) over the source campaign's value: the five imaging RAL CPU legs below were killed at
#: 64 GB inside the gradient measurement (their measured siblings peaked at 38-48 GB, sacct
#: MaxRSS); RAL CPU nodes hold 921-976 GB.
GRADIENT_MEM_GB = {("imaging_p2a", "ral_cpu"): 192}

#: The version-2 run's measured legs (2026-11-01, RAL 413791-413830, jax 0.11.2, 2026.10.25.1, FD
#: step sweep included): cell -> (walls.tsv WALL s, sacct Elapsed s, array task). Every leg wrote a
#: witness valid under gradient_witness.py --check.
P6_WALLS = {
    "cluster/image_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (511, 521, "413791_0"),
    "cluster/image_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (743, 755, "413824_0"),
    "cluster/source_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (76, 85, "413791_1"),
    "cluster/source_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (155, 168, "413824_1"),
    "datacube/delaunay/alma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        1776,
        1783,
        "413811_0",
    ),
    "datacube/delaunay/alma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (333, 343, "413827_0"),
    "datacube/delaunay/alma_high jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        2697,
        2704,
        "413811_1",
    ),
    "datacube/delaunay/alma_high jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        421,
        432,
        "413827_1",
    ),
    "datacube/delaunay/sma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        2020,
        2028,
        "413811_2",
    ),
    "datacube/delaunay/sma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (263, 267, "413827_2"),
    "group/delaunay/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (2447, 2454, "413812_0"),
    "group/delaunay/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (176, 178, "413828_0"),
    "group/mge/euclid jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (136, 143, "413812_1"),
    "group/mge/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (194, 198, "413828_1"),
    "group/rectangular/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (2389, 2397, "413812_2"),
    "group/rectangular/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (161, 164, "413828_2"),
    "imaging/delaunay/ao jax_gpu_dense_fp64@hpc_a100 [gradient]": (265, 276, "413825_0"),
    "imaging/delaunay/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (576, 584, "413792_1"),
    "imaging/delaunay/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (172, 182, "413825_1"),
    "imaging/delaunay/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (1352, 1361, "413792_2"),
    "imaging/delaunay/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (231, 233, "413825_2"),
    "imaging/delaunay/jwst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (2940, 2948, "413792_3"),
    "imaging/delaunay/jwst jax_gpu_dense_fp64@hpc_a100 [gradient]": (237, 239, "413825_3"),
    "imaging/delaunay_nn/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (260, 264, "413825_4"),
    "imaging/mge/ao jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (350, 358, "413792_5"),
    "imaging/mge/ao jax_gpu_dense_fp64@hpc_a100 [gradient]": (146, 149, "413825_5"),
    "imaging/mge/euclid jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (59, 67, "413792_6"),
    "imaging/mge/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (107, 111, "413825_6"),
    "imaging/mge/hst jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (73, 82, "413792_7"),
    "imaging/mge/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (113, 115, "413825_7"),
    "imaging/mge/jwst jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (83, 86, "413792_8"),
    "imaging/mge/jwst jax_gpu_dense_fp64@hpc_a100 [gradient]": (109, 112, "413825_8"),
    "imaging/rectangular/ao jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (15664, 15667, "413792_9"),
    "imaging/rectangular/ao jax_gpu_dense_fp64@hpc_a100 [gradient]": (223, 226, "413825_9"),
    "imaging/rectangular/euclid jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        497,
        499,
        "413792_10",
    ),
    "imaging/rectangular/euclid jax_gpu_dense_fp64@hpc_a100 [gradient]": (147, 150, "413825_10"),
    "imaging/rectangular/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (1290, 1293, "413792_11"),
    "imaging/rectangular/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (217, 220, "413825_11"),
    "imaging/rectangular/jwst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        2902,
        2904,
        "413792_12",
    ),
    "imaging/rectangular/jwst jax_gpu_dense_fp64@hpc_a100 [gradient]": (220, 223, "413825_12"),
    "interferometer/delaunay/alma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        339,
        347,
        "413793_0",
    ),
    "interferometer/delaunay/alma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        71,
        81,
        "413826_0",
    ),
    "interferometer/delaunay/alma_high jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        1106,
        1115,
        "413793_1",
    ),
    "interferometer/delaunay/alma_high jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        102,
        113,
        "413826_1",
    ),
    "interferometer/delaunay/jvla jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        6874,
        6882,
        "413793_2",
    ),
    "interferometer/delaunay/jvla jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        299,
        302,
        "413826_2",
    ),
    "interferometer/delaunay/sma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        377,
        385,
        "413793_3",
    ),
    "interferometer/delaunay/sma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        46,
        48,
        "413826_3",
    ),
    "interferometer/mge/sma jax_cpu_dense_fp64_nufft@hpc_ral_cpu [gradient]": (66, 74, "413793_4"),
    "interferometer/mge/sma jax_gpu_dense_fp64_nufft@hpc_a100 [gradient]": (41, 44, "413826_4"),
    "interferometer/rectangular/alma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        303,
        311,
        "413793_5",
    ),
    "interferometer/rectangular/alma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        42,
        44,
        "413826_5",
    ),
    "interferometer/rectangular/alma_high jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        1050,
        1058,
        "413793_6",
    ),
    "interferometer/rectangular/alma_high jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        78,
        80,
        "413826_6",
    ),
    "interferometer/rectangular/jvla jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        6844,
        6852,
        "413793_7",
    ),
    "interferometer/rectangular/jvla jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        281,
        283,
        "413826_7",
    ),
    "interferometer/rectangular/sma jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu [gradient]": (
        307,
        309,
        "413793_8",
    ),
    "interferometer/rectangular/sma jax_gpu_sparse_fp64_wtilde@hpc_a100 [gradient]": (
        33,
        37,
        "413826_8",
    ),
    "multi_dataset/delaunay/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        2797,
        2804,
        "413811_3",
    ),
    "multi_dataset/delaunay/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (406, 409, "413827_3"),
    "multi_galaxy/delaunay/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        1713,
        1720,
        "413812_3",
    ),
    "multi_galaxy/delaunay/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (180, 184, "413828_3"),
    "multi_galaxy/mge/hst jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (121, 128, "413812_4"),
    "multi_galaxy/mge/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (195, 198, "413828_4"),
    "multi_galaxy/rectangular/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (
        1655,
        1662,
        "413812_5",
    ),
    "multi_galaxy/rectangular/hst jax_gpu_dense_fp64@hpc_a100 [gradient]": (160, 163, "413828_5"),
    "point_source_image/image_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (
        46,
        54,
        "413813_0",
    ),
    "point_source_image/image_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (
        71,
        74,
        "413829_0",
    ),
    "point_source_source/source_plane/simple jax_cpu_na_fp64@hpc_ral_cpu [gradient]": (
        19,
        26,
        "413813_1",
    ),
    "point_source_source/source_plane/simple jax_gpu_na_fp64@hpc_a100 [gradient]": (
        26,
        30,
        "413829_1",
    ),
    "weak/isothermal/simple jax_cpu_dense_fp64@hpc_ral_cpu [gradient]": (14, 21, "413814_0"),
    "weak/isothermal/simple jax_gpu_dense_fp64@hpc_a100 [gradient]": (10, 14, "413830_0"),
}

#: Killed at --mem=192gb (OUT_OF_MEMORY) inside the gradient measurement in the version-2 run (at
#: 64 GB in version 1, 413717_0 / 413717_4): cell -> (Elapsed s, task).
P6_OOM = {
    "imaging/delaunay/ao jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (9502, "413792_0"),
    "imaging/delaunay_nn/hst jax_cpu_sparse_fp64@hpc_ral_cpu [gradient]": (1261, "413792_4"),
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
    """The version-2 run's wall for ``task`` (the FD step sweep included)."""
    wall, elapsed, ref = P6_WALLS[task.cell]
    return max(wall, elapsed), ref


WALL_BASIS = _common.WallBasis(
    measured_wall=measured_wall,
    basis_leg=lambda task: task,
    limitations={},
    walls=P6_WALLS,
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
        "# WALL-BASIS: a measured row cites the version-2 run's wall of its legs (the slowest; array",
        "# tasks; the FD step sweep included). An unmeasured row is a leg that run did not finish: a",
        "# declared guess,",
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
