# Phase 8B / RAL job 340576 — the wall-clock loss, and the rates it yielded

**35 of 39 arms killed at ~12% of budget.** An entire overnight A100 block,
lost to a `--time` justified by the wrong cell.

## What happened

`hpc/batch_gpu/submit_phase8b_bijector_a100` was submitted with
`--time=0:30:00`. Its own header justified that:

> ESTIMATED WALL — 16 starts x 3000 steps at the #117-validated pixelized
> throughput is ~5 min including compile per task (matches the
> diagnostic_theta_e submit's n16/3000-step citation); --time below gives it 6x
> headroom.

The cited `diagnostic_theta_e` throughput is an **MGE** rate. The array it was
being used to size is 20 `delaunay_adapt_split` arms, 15 `knn` arms and 4 `mge`
controls. Only the last group is what the citation described, and only that
group survived.

## The measured rates

Recovered 2026-08-25 from the killed arms themselves — steps completed over
wall elapsed, read from each arm's own log. A truncated arm still measures
s/step, which is the one useful thing a dead block leaves behind.

A100, fp64, 16 lanes, `batch_size=4`, HST:

| cell | s/step | 3000 steps | vs mge | vs the 30 min budget |
|---|---|---|---|---|
| `mge` | 0.117 | ~350 s | 1x | fits, 5x over |
| `knn` | 2.23 | ~1.9 h | 19x | **~8x short** |
| `delaunay_adapt_split` | 4.83 | ~4.0 h | 41x | **~16x short** |

So the claimed "6x headroom" was, for the cells that made up 35 of the 39 arms,
a deficit of roughly one order of magnitude.

## Why the citation could not transfer

Pixelized cells are **per-eval-inversion bound**; parametric cells are not. The
per-step cost of `mge` is dominated by ~25 analytical Gaussians; the per-step
cost of `delaunay_adapt_split` is dominated by solving an inversion on a mesh,
every evaluation. There is no shared term that would make one a proxy for the
other, and the 41x spread is the size of that gap. The same fact shows up
independently in the n_batch scan (#163), where delaunay saturates at 1.26x
while mge gains 1.78x — a pixelized cell has almost nothing left to amortise.

Nothing about a step rate is portable across that boundary, and the confident
prose around the number in the submit header did not make it so.

## What changed as a result

The three rows above are now the seed of `scripts/misc/wall/rates.py`, and
`scripts/misc/wall/check_submits.py` gates every `submit_search_*` /
`submit_phase8b_*` on a `# WALL-BASIS:` block carrying **one row per cell the
submit actually runs**. The cells are read from the job's real
`python3 scripts/.../<cell>.py` invocation, not from its header prose, so an
MGE row can no longer cover a delaunay arm. `--time` is checked against the
**slowest** row. The gate runs in the `lint` workflow.

`submit_phase8b_bijector_a100` itself now sits at `--time=7:00:00`, set by the
delaunay row (4.83 x 3000 + ~300 s compile ~ 4.1 h, x1.5).

Reproduced as a regression test:
`scripts/misc/test/test_wall_check_submits.py::test__phase8b_as_shipped_is_rejected`
reconstructs the submit exactly as it went out and asserts the checker refuses
it.

## The rule

> Never carry a `--time` justification across cells. Derive it from a rate
> measured on *that* cell, and when no measurement exists, run one short arm
> first.

Methodology and the block's grammar: `scripts/misc/wall/README.md`.

## Caveat on the scientific readout

The 340576 arms are **truncated, not failed**. Their partial traces were still
informative — the preliminary bijector signal (`none` stalling at log_post
-154k/-137k while `log_reg`/`logit` reach +20.8k/+28.0k) comes from this block.
They are not a substitute for the completed rerun, and no verdict on F1–F5
should rest on them.
