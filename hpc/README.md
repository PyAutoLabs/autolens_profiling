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
repo; `runtime_` prefixed submits drive `likelihood_runtime/` cells, `nss_`
prefixed ones drive `searches/` cells.

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
