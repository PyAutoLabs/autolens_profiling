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
``output/p6_gradient/`` scratch tree. Nothing has been measured for a gradient leg yet: every
WALL-BASIS row is ``unmeasured`` with ``probe-first: yes`` and a declared guess (4x the cell's
measured phase-2 runtime wall, at least 1200 s; 3x floor). Nothing here submits anything; phase
6c runs them on the human's go, from a session that can reach RAL.
"""

from __future__ import annotations

import argparse
import importlib
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

PIN = "2026.10.25.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#507"
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


def _mem(source: str, kind: str, tasks) -> int:
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
    guess = max(t.guess_s for t in tasks)
    time = _common.slurm_time(guess, headroom=GUESS_HEADROOM, slack_s=0, round_s=WALL_ROUND_S)
    mem = _mem(source, kind, tasks)
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
        "# pre-registered finite-difference check) instead of timing it. It writes gradient.json",
        "# into the mirrored cell named by --output-path under the gitignored output/p6_gradient/",
        "# scratch tree, never in place over committed rows. fp64 only.",
        "#",
        f"# MEMORY. --mem={mem}gb, the phase-2 source campaign's declared value for these cells.",
        "#",
        "# WALL-BASIS: nothing measured for a gradient leg yet. Each row's wall is a declared guess,",
        f"# {GUESS_FACTOR}x the slowest covered cell's measured phase-2 runtime wall (at least "
        f"{GUESS_FLOOR_S} s),",
        "# with the 3x floor; the legs' walls land in output/p6_gradient_logs/walls.tsv.",
    ]
    header += _common.wall_rows(tasks, device, guess, measured=False, sequential=False, basis=None)
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
