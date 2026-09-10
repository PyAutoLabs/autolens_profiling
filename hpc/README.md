# hpc

SLURM submit scripts for the RAL HPC — the `hpc_a100_fp64` / `hpc_a100_mp`
rows of the sweep matrix (and HPC-CPU rows) that the local
`likelihood_runtime/sweep.py` driver cannot run itself.

## Layout

```
batch_gpu/
  submit_<package>_<class>_<model>_a100_<inst>_<precision>[_sparse]   # one submit per cell/config
  output/   error/                                                    # SLURM stdout/stderr (gitignored)
batch_cpu/
  submit_...                                                          # HPC-CPU rows
sync                                                                  # laptop-side driver (below)
sync.conf.example                                                     # template; sync.conf is gitignored
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

## `hpc/sync` — driving RAL from the laptop, and getting runs back

`hpc/sync` is the laptop-side driver. Copy the config first:

```bash
cp hpc/sync.conf.example hpc/sync.conf   # then edit; sync.conf is gitignored
```

| verb | what it does |
|------|--------------|
| `hpc/sync pull` | Download batch logs, then run outputs and results |
| `hpc/sync logs` | Batch logs only — small and fast, use mid-run |
| `hpc/sync status` | Dry run: what a pull would transfer |
| `hpc/sync submit [--gpu\|--cpu] <name>` | `sbatch` a submit script, from `hpc/batch_<type>/` |
| `hpc/sync jobs` / `sacct` / `cancel <id>` | `squeue` / `sacct` / `scancel` |
| `hpc/sync tail [gpu\|cpu]` | Stream the newest live `.out` |
| `hpc/sync du` | Remote disk usage (`-d1` — never a bare recursive `du`; RAL's NFS is slow) |
| `hpc/sync check` | Verify SSH, remote paths, `sbatch`, and the local pull root |

`submit` runs `sbatch` **from `hpc/batch_<type>/`**, not the project root, because the
submit scripts' `#SBATCH -o`/`-e` paths are relative — see the CWD trap below.

### Where a pull lands: `LOCAL_PULL_ROOT`

The pull destination is **not** this checkout. `output/` is gitignored here, `results/`
holds small committed result rows that a staging copy must not overwrite, and a full pull
is ~2 GB of run directories. So `sync.conf` sets `LOCAL_PULL_ROOT` — a separate local
mirror whose only job is holding RAL runs so they can be read. It defaults to
`/mnt/c/Users/Jammy/Science/inference_programme`.

| on RAL | lands at |
|--------|----------|
| `output/searches/` | `$LOCAL_PULL_ROOT/output/searches/` |
| `results/searches/` | `$LOCAL_PULL_ROOT/results/searches/` (a **staging copy** — the committed rows are the ones in this repo) |
| `hpc/batch_gpu/output/` | `$LOCAL_PULL_ROOT/logs/output/` |
| `hpc/batch_gpu/error/` | `$LOCAL_PULL_ROOT/logs/error/` |

`search_internal/` is excluded from every pull: it is sampler state (Nautilus
`checkpoint.hdf5`, live points), it is large, and it is not needed to read a result. A run
that needs its checkpoint is read on RAL.

### The pull manifest: `.cortex/pull.json`

Because `search_internal/` never comes back, a checkpoint that is still growing on RAL is
invisible from the laptop — and "is this run alive?" is exactly what a stalled overnight
sweep needs answered. So a real `pull` ends by writing
`$LOCAL_PULL_ROOT/.cortex/pull.json`: one `find` over the ControlMaster mux records the
size and mtime of every `checkpoint.hdf5` under `output/searches` on RAL, keyed by run
directory relative to the pull root (`output/searches/<…>/<hash>` — the mirror maps 1:1).
PyAutoCortex `collect` reads it to score the checkpoint leg of a run's evidence. The
`runs` map stays empty here: this repo's SLURM logs do not name the run directory a job
wrote, so a job id cannot honestly be linked to a checkpoint. `hpc/sync status` is a dry
run and does not write the manifest, a failed gather writes it anyway with a
`gather_error` key rather than failing the pull, and `.cortex/` is gitignored in the
mirror — it is pull state, like `output/`.

The logs move from `hpc/batch_gpu/{output,error}/` on RAL to `logs/{output,error}/`
locally. On RAL they live beside the submit scripts because that is where SLURM writes
them; on the laptop they are review material, so they sit at the top of the mirror.

`pull` tolerates rsync exit 23/24 (files vanished or partially transferred), which is
normal while jobs are still writing, and skips any remote directory that does not exist
yet rather than aborting.

### There is no `push`

`hpc/sync push` prints the real procedure and exits non-zero. The RAL copy of this repo is
a **git checkout** with local uncommitted state; rsyncing a laptop tree over it would
clobber that and leave the working tree disagreeing with its own HEAD. Code goes to RAL
with git on the login node:

```bash
ssh euclid_jump
cd /mnt/ral/jnightin/autolens_profiling
git status        # look before you pull — the checkout is often dirty
git pull
```

The PyAuto* libraries are a separate story again: resolved from the shared checkouts on
`PYTHONPATH` and updated with `HPCPullPyAuto`, never pip-installed and never rsynced.

## History: the euclid-ral-gpu-1 exclusion (2026-08-28 → 2026-09-05)

**Retired 2026-09-05 (autolens_profiling#220). Both GPU nodes are usable again;
no submit here carries `--exclude` and there is no preflight to source.**

What happened: one A100 on `euclid-ral-gpu-1` (PCI `07:00.0`) was switched into
MIG mode with **no MIG instances created**, while SLURM went on advertising the
node as a plain `Gres=gpu:A100:4`. A job landing on that GPU died about four
seconds in with `RuntimeError: Unable to initialize backend 'cuda'` — and,
because every submit ended in `date`, SLURM recorded it as COMPLETED. Jobs
`341874` / `341875` (2026-08-26) lost 13 tasks that way, all on that node.

The fix (PR #181, 2026-08-28) was `#SBATCH --exclude=euclid-ral-gpu-1` on every
`batch_gpu/` submit plus `hpc/batch_gpu/_gpu_preflight.sh` sourced as a requeue
backstop. SLURM cannot avoid a single GPU inside a node, so it cost 4 of the 8
A100s.

The retirement: on 2026-09-05, from inside
`srun --partition=gpu --nodelist=euclid-ral-gpu-1 --gres=gpu:4`,
`nvidia-smi --query-gpu=index,pci.bus_id,mig.mode.current --format=csv`
reported `Disabled` on all four cards including `07:00.0`, and a JAX CUDA
backend init plus a `jnp` reduction succeeded on each GPU in turn under
`CUDA_VISIBLE_DEVICES=0..3`. A witness `submit_probe_fast_a100` was then
dispatched with `sbatch --nodelist=euclid-ral-gpu-1` and completed: job
`342276` on `euclid-ral-gpu-1`, COMPLETED 0:0 in 22:04, `nvidia-smi` reporting
`MIG M. Disabled`, all 14 probes run, no `!!! probe FAILED` and no traceback in
`error/error.342276.err`.

**If the four-second `cuInit` failure ever returns:** re-probe with that `srun`
command, then re-add `#SBATCH --exclude=<node>` to every submit under
`batch_gpu/` and restore `_gpu_preflight.sh` from git history (it was deleted in
#220). A silent removal re-opens the trap.

## A crashed job FAILS: the SLURM exit-code guard (2026-08-29)

Every submit here ends the same way:

```bash
python3 scripts/... \
    --instrument hst ...

echo "Finished."
date
```

A bash script exits with the status of its **last** command, so that `date`
made every job exit `0` no matter what Python did. RAL job **341988** (the W6
delaunay `n_batch` tail) hit a cuFFT batched plan needing **25.31 GiB** of
scratch, died with `JaxRuntimeError`, printed `Finished.`, and was recorded by
SLURM as **COMPLETED 0:0** in 1:02 and 1:07. `sacct` showed a clean, fast run.
The traceback existed only in `error/error.341988_*.err`, and the only other
tell was a results JSON that never appeared.

The guard now lives in the repo-root **`activate.sh`**, which every submit
already sources:

```bash
if [ -n "${SLURM_JOB_ID:-}" ]; then
    set -eE
    trap '...; exit ${rc}' ERR
fi
```

- **Scoped to a SLURM job.** `activate.sh` is also sourced in interactive
  login-node shells, where `errexit` would close the terminal on the first
  typo. `$SLURM_JOB_ID` is set only inside a job.
- **No `pipefail`.** No submit pipes anything into `python3`, so `pipefail`
  would add risk without covering the failure the guard exists for.
- **Deliberately-tolerated failures are unaffected.** `run_probe` in
  `submit_probe_fast_a100` and `run_cell` in `submit_slogdet_ab_adaptsplit_*`
  already end in `|| echo "!!! ... FAILED"`, and `errexit` does not fire on a
  command whose status is consumed by `||`.
- **Nothing per-submit to remember.** A new submit inherits the guard by
  sourcing `activate.sh`, which it must do anyway to get the venv and
  `PYTHONPATH`.

After this, a Python crash exits the job with Python's status and SLURM records
`FAILED`. Read `sacct` states as real again — but keep reading `.err`: an arm
can still fail *by producing wrong numbers*, which no exit code catches.

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
