"""fnnls warm-start memo after-measurement: reduce the round JSONs (autolens_profiling#426).

``hpc/batch_cpu/submit_memo_after_interferometer_numba_ral_alma_fp64`` runs the #332 ``memo``
lever of the two ALMA numba cells (``delaunay``, ``rectangular``) for six independent rounds at
two iid stream sizes (``n6`` = the exact #332 flags, ``n17`` = 16 iid instances). Each round
writes ``<cell>_numba_hpc_ral_cpu_fp64_memo_after_<size>_r<k>.json`` under ``output/memo_after/``
(gitignored). This script reduces them to one committed summary.

Per cell x stream size x stream (iid, local_walk) it reports:

- the **headline**: ``median(on) / median(off)`` over the pooled timed solves of every round,
  with the paired whole-round bootstrap 90 % interval (``round_bootstrap.round_median_ratio``,
  ``effective_n`` = rounds; #407 / #410 / #412 conventions: median headline, >= 5 rounds);
- ``reading``: where the interval sits relative to 1.0 (``on slower`` when the whole interval is
  above 1, ``on faster`` when it is wholly below, ``includes 1`` otherwise), and
  ``INCONCLUSIVE`` below :data:`ab_verdict.MIN_AB_ROUNDS` rounds. It is a description of the
  measurement, not a verdict on PyAutoArray#613;
- the #332 estimator beside it (``on_over_off_mean`` per round, its median and range), so the
  number compares like with like against the 2.17x;
- the slow-seed count: timed on-solves slower than 1.5x the round's off median;
- the figure-of-merit agreement (max |on - off| nats) and the host / load per round.

The first solve of every pass has nothing to seed from and is excluded, as in #332.

    python scripts/interferometer/delaunay/memo_after_summary.py \\
        --rounds-dir output/memo_after --version 2026.10.17.1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))

from likelihood_breakdown.ab_verdict import MIN_AB_ROUNDS  # noqa: E402
from likelihood_breakdown.round_bootstrap import round_median_ratio  # noqa: E402

NAME = re.compile(
    r"^(?P<cell>delaunay|pixelization|rectangular)_numba_hpc_ral_cpu_fp64_memo_after_"
    r"(?P<size>n6|n17)_r(?P<round>\d+)\.json$"
)
BASELINE = {
    "delaunay": "results/breakdown/interferometer/delaunay_numba_hpc_ral_cpu_fp64_levers.json",
    "rectangular": (
        "results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_levers.json"
    ),
}
SLOW_SEED_FACTOR = 1.5
SEED = 426


def _reading(ci_low: float, ci_high: float, n_rounds: int) -> str:
    if n_rounds < MIN_AB_ROUNDS:
        return f"INCONCLUSIVE ({n_rounds} < {MIN_AB_ROUNDS} rounds)"
    if ci_low > 1.0:
        return "on slower (interval wholly above 1)"
    if ci_high < 1.0:
        return "on faster (interval wholly below 1)"
    return "includes 1"


def _stream_summary(rounds: list[dict], stream: str) -> dict:
    on = np.array([r["levers"]["memo"][stream]["on"]["per_instance_s"][1:] for r in rounds])
    off = np.array([r["levers"]["memo"][stream]["off"]["per_instance_s"][1:] for r in rounds])
    head = round_median_ratio(on, off, n_rounds=len(rounds), seed=SEED)
    per_round_mean = [r["levers"]["memo"][stream]["on_over_off_mean"] for r in rounds]
    slow = int(sum((on[i] > SLOW_SEED_FACTOR * np.median(off[i])).sum() for i in range(len(on))))
    return {
        "headline_on_over_off_median": head["ratio"],
        "ci90": [head["ci90_low"], head["ci90_high"]],
        "resampling": head["resampling"],
        "effective_n_rounds": head["effective_n"],
        "bootstrap_samples": head["bootstrap_samples"],
        "reading": _reading(head["ci90_low"], head["ci90_high"], head["effective_n"]),
        "median_on_ms": float(np.median(on) * 1e3),
        "median_off_ms": float(np.median(off) * 1e3),
        "timed_solves_per_round": int(on.shape[1]),
        "on_over_off_mean_per_round": per_round_mean,
        "on_over_off_mean_median_of_rounds": float(np.median(per_round_mean)),
        "on_over_off_mean_range": [float(min(per_round_mean)), float(max(per_round_mean))],
        "slow_seed_on_solves": slow,
        "slow_seed_rule": f"on-solve > {SLOW_SEED_FACTOR} x its round's off median",
        "on_solves_total": int(on.size),
        "max_abs_figure_of_merit_diff_nats": float(
            max(r["levers"]["memo"][stream]["max_abs_figure_of_merit_diff_nats"] for r in rounds)
        ),
        "per_round_on_s": on.tolist(),
        "per_round_off_s": off.tolist(),
    }


def _baseline(cell: str) -> dict:
    d = json.loads((ROOT / BASELINE[cell]).read_text())
    m = d["levers"]["memo"]
    return {
        "path": BASELINE[cell],
        "autolens_version": d.get("autolens_version"),
        "PyAutoArray": d.get("source_revisions", {}).get("PyAutoArray"),
        "host": d.get("device", {}).get("hostname"),
        "rounds": 1,
        "iid_on_over_off_mean": m["iid"]["on_over_off_mean"],
        "iid_timed_solves": m["iid"]["off"]["n"],
        "local_walk_on_over_off_mean": m["local_walk"]["on_over_off_mean"],
        "local_walk_timed_solves": m["local_walk"]["off"]["n"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rounds-dir", default=str(ROOT / "output" / "memo_after"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--job", default=None, help="SLURM job id, for provenance.")
    parser.add_argument(
        "--out",
        default=None,
        help="Default results/breakdown/interferometer/memo_after_hpc_ral_cpu_fp64_v<version>.json",
    )
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for path in sorted(Path(args.rounds_dir).glob("*.json")):
        m = NAME.match(path.name)
        if not m:
            continue
        cell = "rectangular" if m["cell"] == "pixelization" else m["cell"]
        groups.setdefault((cell, m["size"]), []).append(
            (int(m["round"]), json.loads(path.read_text()))
        )
    if not groups:
        raise SystemExit(f"no round JSONs under {args.rounds_dir}")

    cells: dict = {}
    for (cell, size), items in sorted(groups.items()):
        items.sort(key=lambda t: t[0])
        rounds = [d for _, d in items]
        first = rounds[0]
        entry = {
            "rounds": [k for k, _ in items],
            "n_instances_flag": first.get("configuration", {}).get("n_instances"),
            "memo_clear_method": first["levers"]["memo"].get("memo_clear_method"),
            "walk_unit_step": first["levers"]["memo"].get("walk_unit_step"),
            "hosts": sorted({str(d.get("configuration", {}).get("hostname")) for d in rounds}),
            "load_avg_start_end_per_round": [
                [
                    d.get("configuration", {}).get("host_load_avg_start"),
                    d.get("configuration", {}).get("host_load_avg_end"),
                ]
                for d in rounds
            ],
            "source_revisions": first.get("source_revisions"),
            "configuration": {
                k: first.get("configuration", {}).get(k)
                for k in (
                    "visibilities",
                    "source_pixels",
                    "mesh",
                    "mask_radius_arcsec",
                    "inversion_class",
                    "headline_arm",
                    "numba_version",
                )
            },
        }
        for stream in ("iid", "local_walk"):
            entry[stream] = _stream_summary(rounds, stream)
        cells.setdefault(cell, {})[size] = entry

    payload = {
        "cell": "interferometer/{delaunay,rectangular}_numba/alma levers.memo, after PyAutoArray#615",
        "issue": "autolens_profiling#426",
        "release_version": args.version,
        "slurm_job": args.job,
        "question": (
            "fnnls warm-start memo on vs off, reconstruction (solve) alone, iid and local-walk "
            "streams: the #332 lever re-measured on the released stack"
        ),
        "estimators": (
            "headline = median(on)/median(off) over pooled timed solves, paired whole-round "
            "bootstrap 90 % interval; #332 estimator = mean(on)/mean(off) per round"
        ),
        "baseline_332": {cell: _baseline(cell) for cell in cells},
        "cells": cells,
    }
    out = (
        Path(args.out)
        if args.out
        else (
            ROOT
            / "results"
            / "breakdown"
            / "interferometer"
            / f"memo_after_hpc_ral_cpu_fp64_v{args.version}.json"
        )
    )
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for cell, sizes in cells.items():
        for size, e in sizes.items():
            for stream in ("iid", "local_walk"):
                s = e[stream]
                print(
                    f"{cell:12s} {size:4s} {stream:10s} median on/off {s['headline_on_over_off_median']:.3f} "
                    f"[{s['ci90'][0]:.3f}, {s['ci90'][1]:.3f}] ({s['reading']}); "
                    f"#332-mean median {s['on_over_off_mean_median_of_rounds']:.3f} "
                    f"range {s['on_over_off_mean_range'][0]:.2f}-{s['on_over_off_mean_range'][1]:.2f}; "
                    f"slow seeds {s['slow_seed_on_solves']}/{s['on_solves_total']}"
                )


if __name__ == "__main__":
    main()
