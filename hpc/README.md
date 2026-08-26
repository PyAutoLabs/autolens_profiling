# hpc

SLURM submit scripts for the RAL HPC — the `hpc_a100_fp64` / `hpc_a100_mp`
rows of the sweep matrix (and HPC-CPU rows) that the local
`likelihood_runtime/sweep.py` driver cannot run itself.

## Layout

```
batch_gpu/
  submit_<package>_<class>_<model>_a100_<inst>_<precision>[_sparse]   # one submit per cell/config
  output/   error/                                                    # SLURM stdout/stderr (gitignored)
```

Submit names follow the same `<class>/<model>` cell grid as the rest of the
repo; `runtime_` prefixed submits drive `likelihood_runtime/` cells.

## Running

On the HPC login node:

```bash
source activate.sh          # repo-root helper: venv + PYTHONPATH at the canonical checkouts
sbatch hpc/batch_gpu/submit_runtime_imaging_mge_a100_hst_fp64
```

Each job writes its per-config JSON into the same `results/` layout as a local
sweep (`--config-name hpc_a100_fp64` etc.), so `likelihood_runtime/aggregate.py`
merges local and A100 rows into one `comparison.json`. Copy/commit the result
JSONs from the HPC checkout back via the normal git flow. The PyAuto*
libraries resolve from sibling source checkouts on `PYTHONPATH` — never
pip-install them into the venv (`HPCPullPyAuto` is the update story).

## Array submits (repeated-seed campaigns)

A submit whose name ends in a tier (e.g. `..._fp64_n64`) and that declares
`#SBATCH --array=0-4` runs one **seed per array task** — the shape a
reliability measurement needs (`results/notes/inference/PROGRAMME.md` §3:
"reliability is P(correct | fixed budget), measured over >= 5 seeds"). The
seed is read from a `SEEDS=(...)` bash array indexed by `SLURM_ARRAY_TASK_ID`
and exported as `SEARCHES_SEED`; stdout/stderr use the `%A_%a` (job_array)
pattern so each task's log is separate.

## Wall clock: `# WALL-BASIS:` is mandatory on searches submits

Every `submit_search_*` / `submit_phase8b_*` must carry a `# WALL-BASIS:` block
above its `#SBATCH` stanza, with **one row per cell it runs**, and
`scripts/misc/wall/check_submits.py --check` gates it on every PR.

The rule the block enforces: **never carry a `--time` justification across
cells.** `submit_phase8b_bijector_a100` cited an MGE step rate for an array of
mostly `knn` and `delaunay_adapt_split` arms; at 16 lanes / `batch_size=4` on
A100 fp64 those run 19x and 41x slower than mge, and RAL job 340576 lost 35 of
its 39 arms at ~12% of budget — an entire overnight A100 block. An array submit
is sized by its **slowest** cell, never its fastest.

Measured rates live in `scripts/misc/wall/rates.py`; the block's grammar, the
three `source:` kinds (`rates` / `measured-wall` / `unmeasured`) and how to add
a measured row are in `scripts/misc/wall/README.md`. When a cell has no
measurement, say so with `source: unmeasured  probe-first: yes` and run one
short arm — a truncated arm still measures s/step.

Every arm must land in its own results file **and** its own autofit output
directory. `--config-name` carries the tier and seed, which makes the results
filename distinct; the search-side half is handled by
`searches/_samplers.multi_start_unique_tag` — read its docstring before adding
an arm, because only `clipper` enters a MultiStart search's identifier and an
untagged arm silently returns a sibling arm's completed fit.
