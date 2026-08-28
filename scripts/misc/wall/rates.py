"""Per-cell measured step rates for HPC submit `--time` estimates.

Populated **only** from rates measured on the cell the row names. Nothing in
this table is interpolated, extrapolated, or carried over from a neighbouring
cell — that carry-over is the defect this module exists to prevent.

Why it exists
-------------

`hpc/batch_gpu/submit_phase8b_bijector_a100` set ``--time=0:30:00`` and
justified it in its own comment with an **MGE** step rate, for an array whose
arms were mostly ``knn`` and ``delaunay_adapt_split``. Rates measured on RAL
2026-08-25 (from the truncated arms of job 340576):

===================== ========== ==================
cell                  s/step     3000 steps
===================== ========== ==================
mge                   0.117      ~350 s
knn                   2.20       ~1.8 h  (19x)
delaunay_adapt_split  4.85       ~4.0 h  (41x)
===================== ========== ==================

The knn / delaunay_adapt_split rows were re-measured 2026-08-27 from the
COMPLETED arms of jobs 341874/341875 and moved by 1.4% and 0.4% respectively —
the truncated-arm recovery above was sound. The delaunay row quotes its
SLOWEST arm: its full walls span 2.18-4.85 s/step with the spread tracking
``log_det_method`` / ``bijector``, neither of which is in the key.

So the "6x headroom" the comment claimed was ~8x short for knn and ~16x short
for delaunay. **35 of 39 arms were killed at ~12% of budget**, losing an
overnight A100 block; the only 4 that finished were the mge controls — the
cells the citation actually described.

Pixelized cells are per-eval-inversion bound and parametric cells are not.
Nothing about a step rate is portable across that boundary.

The rule
--------

Never carry a ``--time`` justification across cells. Derive it from a measured
rate for *that* cell, and when no measurement exists, run one short arm first —
a 30-minute truncated arm still measures s/step, which is exactly how the
numbers above were recovered from the failed block.

`wall/check_submits.py` enforces this on every submit that runs a searches
cell; `hpc/README.md` states the authoring contract.

Table shape
-----------

Keys are ``(dataset, cell, instrument, device, precision, n_lanes,
batch_size)``. ``batch_size`` is part of the key — not a detail — because the
unbatched MGE lane rows and the ``batch_size=4`` pixelized rows are genuinely
different configurations of the same cell, and their rates differ by more than
2x at the same lane count. ``None`` means the arm ran unbatched.

Adding a row
------------

1. Measure it on the cell itself — a full arm, or a truncated one (steps
   completed / wall elapsed, from the arm's own log).
2. Add the row below with an inline comment naming the job and date.
3. Add or extend the matching ``PROVENANCE`` entry.
4. Point the submit's ``# WALL-BASIS:`` block at it with ``source: rates``.
"""

from __future__ import annotations

# Key: (dataset, cell, instrument, device, precision, n_lanes, batch_size)
# Value: seconds per step, measured on that exact configuration.
STEP_RATE: dict[tuple[str, str, str, str, str, int, int | None], float] = {
    # =========================================================================
    # Pixelized imaging cells, 16 lanes, batch_size=4 (the mandatory chunking
    # on pixelized cells). Measured 2026-08-25 from the truncated arms of RAL
    # job 340576 — the block that phase8b's MGE citation killed.
    # =========================================================================
    # Both rows were re-measured 2026-08-27 from the COMPLETED 3000-step arms of
    # RAL jobs 341874 / 341875 — full walls, not the truncated arms of 340576 —
    # and the truncated estimates survived: knn 2.23 -> 2.20 (1.4%),
    # delaunay_adapt_split 4.83 -> 4.85 (0.4%). See PROVENANCE
    # `pixelized_hst_a100_fp64_n16_b4_full_arms` for the per-arm walls.
    #
    # delaunay_adapt_split is quoted at its SLOWEST measured arm on purpose.
    # Its full arms span 6531-14540 s (2.18-4.85 s/step) and the spread is not
    # noise: it tracks `log_det_method` and `bijector`, neither of which is in
    # this table's key. An array submit is sized by its slowest cell, and a
    # single-valued row for a cell whose rate varies 2.2x must quote the slow
    # end or it is a budget that kills arms. Do NOT read this row as "what a
    # delaunay arm costs" — read it as "what the slowest one cost".
    ("imaging", "delaunay_adapt_split", "hst", "a100", "fp64", 16, 4): 4.85,  # 3000 steps ~ 4.0 h
    ("imaging", "knn", "hst", "a100", "fp64", 16, 4): 2.20,  # 3000 steps ~ 1.8 h
    #
    # The MGE control arms from the same job — the 4 of 39 that completed.
    # 41x faster than delaunay_adapt_split at the SAME lanes/batch_size.
    ("imaging", "mge", "hst", "a100", "fp64", 16, 4): 0.117,  # 3000 steps ~ 350 s
    #
    # =========================================================================
    # Parametric imaging cell, unbatched, per lane tier. These are the rows the
    # n16/n64/n256 multi_start_prodigy submits already cite in their own
    # ESTIMATED WALL blocks, measured on A100 fp64 with a warm compile cache.
    # They are MGE-only and must never be quoted for a pixelized cell.
    # =========================================================================
    ("imaging", "mge", "hst", "a100", "fp64", 16, None): 0.05,  # ~150 s at the 3000-step ceiling
    ("imaging", "mge", "hst", "a100", "fp64", 64, None): 0.19,  # ~570 s
    ("imaging", "mge", "hst", "a100", "fp64", 256, None): 0.77,  # ~2300 s
    #
    # NOTE — the n128 tier is deliberately ABSENT. Its submit's ~0.38 s/step is
    # interpolated between the n64 and n256 rows, not measured. An interpolated
    # rate is exactly the kind of unearned citation this table refuses to carry;
    # that submit declares `source: measured-wall` against its observed runs
    # instead.
}

PROVENANCE: dict[str, str] = {
    "pixelized_hst_a100_fp64_n16_b4": (
        "measured 2026-08-25 on RAL A100 (job 340576) from the truncated arms of the "
        "Phase 8B bijector A/B; steps completed / wall elapsed per arm. 35 of 39 arms "
        "were killed at ~12% of a 0:30:00 budget set from an MGE citation — these are "
        "the rates recovered from that failure. Write-up: "
        "results/notes/inference/phase_08_regularization/wall_clock_340576.md"
    ),
    "pixelized_hst_a100_fp64_n16_b4_full_arms": (
        "re-measured 2026-08-27 on RAL A100 from the COMPLETED 3000-step arms of jobs "
        "341874 / 341875 (Phase 8B bijector A/B, 16 lanes / batch_size=4). knn: "
        "6530-6600 s total wall across arms -> 2.18-2.20 s/step. delaunay_adapt_split: "
        "6531-14540 s -> 2.18-4.85 s/step, and the spread is structured, not noise — "
        "cholesky arms 3.6-4.0 h, slogdet+log_reg 3.1-3.7 h, slogdet+none 1.8 h. "
        "log_det_method and bijector are NOT in this table's key, so the row quotes the "
        "slowest arm; anything else would under-size an array sized by its slowest cell. "
        "These full walls confirm the 2026-08-25 truncated-arm rates (340576) to within "
        "1.4% (knn) and 0.4% (delaunay_adapt_split) — the recovery method was sound."
    ),
    "mge_hst_a100_fp64_unbatched": (
        "A100 fp64 with a warm compile cache, as cited by the n16/n64/n256 "
        "multi_start_prodigy submits' own ESTIMATED WALL blocks. MGE only."
    ),
}


class UnmeasuredCellError(KeyError):
    """No measured step rate exists for the requested configuration.

    Raised instead of returning a nearby cell's rate. Falling back across cells
    is the bug — see this module's docstring.
    """


def step_rate_for(
    dataset: str,
    cell: str,
    instrument: str,
    device: str,
    precision: str,
    n_lanes: int,
    batch_size: int | None = None,
) -> float:
    """Seconds per step for exactly this configuration.

    There is **no** nearest-neighbour fallback: an unmeasured configuration
    raises `UnmeasuredCellError` rather than silently answering with a rate
    measured on a different cell, lane count or batching.
    """
    key = (dataset, cell, instrument, device, precision, n_lanes, batch_size)
    try:
        return STEP_RATE[key]
    except KeyError:
        raise UnmeasuredCellError(
            f"no measured step rate for {key!r}. Do not substitute another cell's rate — "
            f"run one short arm on this cell and add the row to wall/rates.py, or declare "
            f"`source: unmeasured` with `probe-first: yes` in the submit's WALL-BASIS block."
        ) from None


def wall_estimate(rate_s_per_step: float, n_steps: int, compile_s: float = 0.0) -> float:
    """Estimated wall seconds for `n_steps` at `rate_s_per_step`, plus compile."""
    return rate_s_per_step * n_steps + compile_s
