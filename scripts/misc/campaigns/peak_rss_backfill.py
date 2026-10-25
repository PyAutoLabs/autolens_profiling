"""Phase 2a imaging rows: their peak memory and cores per call, from the RAL jobs' sacct.

The 2a rows (autolens_profiling#456) were measured before the row writers recorded
``resources`` (#462). This module attaches ``cpus_per_eval`` and ``peak_rss_gb`` to each
committed batch-1 row through the sidecar ``catalogue/peak_rss_backfill.json``; rows are never
edited. ``build_catalogue.py`` reads the sidecar for a row that has no ``resources`` block::

    python scripts/misc/campaigns/peak_rss_backfill.py --sacct <dump> --write

``<dump>`` is ``sacct -P -o JobID,JobName,State,Elapsed,MaxRSS,AllocCPUS,NodeList`` over the
2a jobs, then a ``@@WALLS`` line, then the campaign's ``output/p2a_logs/walls.tsv``.

Mapping (``imaging_p2a.py`` is the work list):

- **RAL CPU arrays** (:data:`BATCH1_RAL_ARRAYS`, one leg per task): task ``<array>_<i>`` ran
  ``batch1_ral_cpu(instrument)[i]``. Its ``.batch`` step MaxRSS is that leg's peak:
  ``basis: sacct_maxrss``.
- **A100 jobs** (:data:`BATCH1_A100_JOBS`, every leg of one instrument in turn): only a
  job-level MaxRSS exists. Each leg gets it as an upper bound: ``basis: job_maxrss``.
- ``walls.tsv`` cross-checks both: each leg's ``WALL`` line names the cell at that index.
- A ``.batch`` step whose MaxRSS sacct left empty (a task shorter than the accounting sample)
  gets ``peak_rss_gb: null`` with the reason.
- Batch 0 (one leg per task on ao, and on hst for delaunay_nn; ``imaging_p2a.BATCH0_WALLS``)
  ran the same cell on the same release; its per-task MaxRSS is kept as ``supporting_batch0``,
  never as the row's value.

``cpus_per_eval`` follows the submits: numba 1; JAX on CPU ``NPROC`` = the task's AllocCPUS;
JAX on GPU one device with ``NPROC`` host threads. GB here are 1e9 bytes; sacct's ``K`` is KiB.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIDECAR = "catalogue/peak_rss_backfill.json"

#: Batch 1's RAL CPU arrays (one cell per task) and A100 jobs (every leg of one instrument).
BATCH1_RAL_ARRAYS = {"ao": "412923", "euclid": "412924", "hst": "412925", "jwst": "412926"}
BATCH1_A100_JOBS = {"ao": "412927", "euclid": "412943", "hst": "412944", "jwst": "412945"}

_UNITS_KIB = {"K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}


def _p2a():
    spec = importlib.util.spec_from_file_location(
        "imaging_p2a", Path(__file__).with_name("imaging_p2a.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("imaging_p2a", module)
    spec.loader.exec_module(module)
    return module


def maxrss_kib(raw: str) -> float | None:
    """sacct MaxRSS (``12727812K``, ``25334M``, ``1.5G``, bare bytes) in KiB; None when empty."""
    raw = raw.strip()
    if not raw:
        return None
    if raw[-1] in _UNITS_KIB:
        return float(raw[:-1]) * _UNITS_KIB[raw[-1]]
    return float(raw) / 1024


def gb(kib: float) -> float:
    return round(kib * 1024 / 1e9, 3)


def parse_dump(text: str) -> tuple[dict, list[tuple]]:
    """({job: {state, max_rss_raw, alloc_cpus, node}}, walls) from the sacct + walls dump.

    ``job`` is the allocation (``412923_3`` or ``412927``); its values come from the ``.batch``
    step, except ``state``, which is the allocation's. ``walls`` are ``(kind, job, index,
    value, cell)`` tuples.
    """
    sacct_part, _, walls_part = text.partition("@@WALLS")
    jobs = {}
    for line in sacct_part.splitlines():
        fields = line.split("|")
        if len(fields) < 7 or fields[0] == "JobID":
            continue
        job_id, _name, state, _elapsed, rss, cpus, node = fields[:7]
        base, _, step = job_id.partition(".")
        entry = jobs.setdefault(base, {})
        if not step:
            entry["state"] = state
        elif step == "batch":
            entry.update(max_rss_raw=rss, alloc_cpus=int(cpus), node=node)
    walls = []
    for line in walls_part.splitlines():
        fields = line.split("\t")
        if len(fields) == 5 and fields[0] in ("START", "WALL"):
            walls.append((fields[0], fields[1], int(fields[2]), fields[3], fields[4]))
    return jobs, walls


def _cpus(task, alloc: int) -> dict:
    if task.backend == "numba":
        return {
            "cpus_per_eval": 1,
            "devices_per_eval": 0,
            "cpus_per_eval_basis": "numba: one serial evaluation, numba and BLAS threads pinned to 1",
        }
    if task.device == "gpu":
        return {
            "cpus_per_eval": alloc,
            "devices_per_eval": 1,
            "cpus_per_eval_basis": f"jax gpu: one device; NPROC = AllocCPUS = {alloc} host threads",
        }
    return {
        "cpus_per_eval": alloc,
        "devices_per_eval": 0,
        "cpus_per_eval_basis": f"jax cpu: NPROC = AllocCPUS = {alloc} XLA intra-op threads",
    }


def _peak(job: str, info: dict, basis: str) -> dict:
    kib = maxrss_kib(info.get("max_rss_raw", ""))
    out = {
        "sacct_job": job,
        "sacct_state": info.get("state"),
        "max_rss_raw": info.get("max_rss_raw") or None,
        "peak_rss_gb": None if kib is None else gb(kib),
        "peak_rss_basis": None if kib is None else basis,
    }
    if kib is None:
        out["peak_rss_reason"] = (
            "sacct recorded no MaxRSS for the .batch step (the task ended before the "
            "accounting sample)"
        )
    return out


def build(text: str, exists=lambda path: (ROOT / path).is_file()) -> dict:
    """The sidecar document: one entry per committed batch-1 row, keyed by its path."""
    p2a = _p2a()
    jobs, walls = parse_dump(text)
    wall_cells = {(job, index): cell for kind, job, index, _v, cell in walls if kind == "WALL"}
    wall_by_index = {}
    for kind, _job, index, _v, cell in walls:
        if kind == "WALL":
            wall_by_index.setdefault(index, set()).add(cell)

    def supporting(task) -> dict | None:
        found = p2a.BATCH0_WALLS.get(task.cell)
        if found is None:
            return None
        job = found[2]
        info = jobs.get(job)
        if info is None:
            return None
        kib = maxrss_kib(info.get("max_rss_raw", ""))
        return {
            "sacct_job": job,
            "max_rss_raw": info.get("max_rss_raw") or None,
            "peak_rss_gb": None if kib is None else gb(kib),
            "basis": "sacct_maxrss (batch 0, one leg per task, same cell and release)",
        }

    entries, without_row = {}, []
    legs = []
    for instrument, array in BATCH1_RAL_ARRAYS.items():
        for index, task in enumerate(p2a.batch1_ral_cpu(instrument)):
            job = f"{array}_{index}"
            if task.cell not in wall_by_index.get(index, set()):
                raise ValueError(f"walls.tsv has no WALL for {task.cell} at index {index}")
            legs.append((task, job, "sacct_maxrss"))
    for instrument, job in BATCH1_A100_JOBS.items():
        for index, task in enumerate(p2a.batch1_a100(instrument)):
            if wall_cells.get((job, index)) != task.cell:
                raise ValueError(f"walls.tsv: job {job} index {index} is not {task.cell}")
            legs.append((task, job, "job_maxrss"))
    for task, job, basis in legs:
        if job not in jobs:
            raise ValueError(f"sacct has no job {job}")
        if not exists(task.output):
            without_row.append(task.cell)
            continue
        info = jobs[job]
        entry = {"cell": task.cell, **_cpus(task, info["alloc_cpus"]), **_peak(job, info, basis)}
        if basis == "job_maxrss":
            entry["peak_rss_note"] = (
                "multi-leg A100 job: job-level MaxRSS over all its legs, an upper bound on this leg"
            )
        backing = supporting(task)
        if backing is not None:
            entry["supporting_batch0"] = backing
        entries[task.output] = entry
    return {
        "schema": "peak-rss-backfill",
        "version": 1,
        "campaign": "likelihood-lifecycle phase 2a imaging (autolens_profiling#456), batch 1",
        "note": (
            "Peak resident memory and cores per call for the committed 2a rows, which predate the "
            "row writers' resources block (#462). Generated by "
            "scripts/misc/campaigns/peak_rss_backfill.py from sacct (.batch step MaxRSS, KiB; "
            "GB = 1e9 bytes) and walls.tsv. basis sacct_maxrss: one leg per RAL array task, "
            "exact per leg; basis job_maxrss: multi-leg A100 job, the job-level peak as an upper "
            "bound on each leg; null: sacct sampled no MaxRSS. supporting_batch0 is batch 0's "
            "per-task MaxRSS of the same cell, supporting evidence only."
        ),
        "jobs": {
            "ral_cpu_arrays": BATCH1_RAL_ARRAYS,
            "a100_jobs": BATCH1_A100_JOBS,
        },
        "legs_without_row": sorted(without_row),
        "rows": dict(sorted(entries.items())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sacct", type=Path, required=True, help="sacct + @@WALLS dump")
    parser.add_argument("--write", action="store_true", help=f"write {SIDECAR}")
    args = parser.parse_args(argv)
    doc = build(args.sacct.read_text())
    text = json.dumps(doc, indent=1) + "\n"
    if args.write:
        (ROOT / SIDECAR).write_text(text)
    rows = doc["rows"].values()
    count = {
        basis: sum(r["peak_rss_basis"] == basis for r in rows)
        for basis in ("sacct_maxrss", "job_maxrss", None)
    }
    print(
        f"{SIDECAR}: {len(doc['rows'])} rows (per-leg {count['sacct_maxrss']}, job-level "
        f"{count['job_maxrss']}, none {count[None]}); {len(doc['legs_without_row'])} legs "
        "without a row" + ("" if args.write else " (dry run; --write to save)")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
