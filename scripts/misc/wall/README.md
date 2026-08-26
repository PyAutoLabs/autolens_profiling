# `wall` — per-cell step rates + the submit `--time` gate

This subpackage owns one rule:

> **Never carry a `--time` justification across cells.** Derive it from a rate
> measured on *that* cell. When no measurement exists, run one short arm first.

Two responsibilities:

1. **Rates** (`wall.rates`) — the curated per-cell step-rate table, populated
   only from rates measured on the cell each row names.
2. **Gate** (`wall.check_submits`) — the CI check that every cell a submit
   actually runs has its own basis row, that cited rates match the table, and
   that `--time` clears the declared headroom.

## Why it exists

`hpc/batch_gpu/submit_phase8b_bijector_a100` set `--time=0:30:00` and justified
it in its own comment as "16 starts x 3000 steps at the #117-validated
pixelized throughput is ~5 min including compile per task ... `--time` below
gives it 6x headroom."

That was an **MGE** rate. Measured on RAL 2026-08-25, at 16 lanes /
`batch_size=4` on A100 fp64:

| cell | s/step | 3000 steps | vs mge |
|---|---|---|---|
| `mge` | 0.117 | ~350 s | 1x — the citation was right, *for MGE* |
| `knn` | 2.23 | ~1.9 h | 19x |
| `delaunay_adapt_split` | 4.83 | ~4.0 h | 41x |

The claimed 6x headroom was ~8x short for knn and ~16x short for delaunay.
**35 of 39 arms in job 340576 were killed at ~12% of budget**, losing an
overnight A100 block. The 4 that finished were the mge controls — the only
cells the citation actually described.

Full write-up of the loss and how the rates were recovered from it:
[`results/notes/inference/phase_08_regularization/wall_clock_340576.md`](../../../results/notes/inference/phase_08_regularization/wall_clock_340576.md).

Pixelized cells are per-eval-inversion bound; parametric cells are not. Nothing
about a step rate is portable across that boundary, and no amount of confident
prose around the number makes it so. Hence a table that refuses to answer for a
cell it has not measured, and a gate that reads what the job will really run
rather than what its header claims.

## The `# WALL-BASIS:` block

One row per cell the submit runs, in the header above the `#SBATCH` stanza:

```
# WALL-BASIS: — one row per cell this submit runs.
#   cell: imaging/delaunay_adapt_split/hst  device: a100  precision: fp64
#   lanes: 16  batch_size: 4  steps: 3000  rate: 4.83  source: rates
#   compile: 300  headroom: 1.5
```

A row starts at its `cell:` key and runs to the next `cell:` or the end of the
block. Prose lines inside the block are ignored, so the human "why" can sit
next to the machine-checked "what".

### The three `source:` kinds

| `source:` | needs | headroom floor | means |
|---|---|---|---|
| `rates` | `lanes`, `batch_size`, `steps`, `rate`, `device`, `precision` | 1.5x | a step rate measured on **this** cell, matching `rates.py` within 5% |
| `measured-wall` | `wall` (seconds), `ref` | 1.25x | a directly observed total for this cell — a fork row, or prior runs of this same arm |
| `unmeasured` | `probe-first: yes` | 3x on any `wall` offered | nothing measured. Legal, and honest |

`unmeasured` is deliberately permitted. A legacy submit should not have to
invent a number to satisfy a linter — inventing numbers is the disease. The row
earns its place by forcing the author to state, **per cell**, that this cell's
wall clock rests on nothing. That is exactly what phase8b's prose concealed.

### What the gate checks

1. **Every cell the submit runs has its own row.** The cells are read from the
   `python3 scripts/<dataset>/.../<cell>.py` invocation — resolving
   `${CELLS[$I]}` arrays, `case` arms and variable indirection, and ignoring
   commands merely *mentioned* in comments. This is the check that catches the
   phase8b defect: a cell with no row of its own is a cell whose `--time` was
   justified by some other cell's rate.
2. **A declared cell is one the submit actually runs** (and its instrument).
3. **A `source: rates` row matches `rates.py`** within 5%.
4. **`--time` >= headroom x estimated wall**, over the *slowest* row. An array
   submit is sized by its slowest cell, never its fastest.

### Which submits must carry one

Required on `submit_search_*` and `submit_phase8b_*` — the multi-cell array
submits where a cross-cell mis-citation is possible at all. Validated wherever
else it appears.

This is a **path predicate, not an allowlist**. No submit is individually
exempted, because an exemption list would hide precisely the class of leak this
gate exists to close. Single-cell `submit_runtime_*` / `submit_breakdown_*`
submits run one cell and so cannot mis-cite across cells; forcing a vacuous
`unmeasured` header onto them would add noise, not safety.

## Running it

```bash
python scripts/misc/wall/check_submits.py           # report on every submit
python scripts/misc/wall/check_submits.py --check   # exit non-zero on any violation
```

`--check` runs in the `lint` workflow on every PR and push to main, alongside
`build_readme.py --check`.

## Adding a measured rate

1. **Measure it on the cell itself.** A full arm, or a truncated one — steps
   completed / wall elapsed, read from the arm's own log. A 30-minute killed
   arm still measures s/step, which is exactly how the three rows above were
   recovered from the block that died.
2. Add the row to `STEP_RATE` in `rates.py` with an inline comment naming the
   job and date. The key is
   `(dataset, cell, instrument, device, precision, n_lanes, batch_size)` —
   `batch_size` is in the key because the unbatched lane rows and the
   `batch_size=4` pixelized rows are different configurations of the same cell.
3. Add or extend the matching `PROVENANCE` entry.
4. Point the submit's row at it with `source: rates`.

**Do not interpolate.** The n128 MGE tier is deliberately absent from the table
for this reason: its submit's ~0.38 s/step sits between the measured n64 and
n256 rows but was never measured, so that submit declares `measured-wall`
against its observed runs instead. An interpolated rate is the same unearned
citation in a more respectable coat.
