# results/sweep

Provenance for the multi-leg HPC sweeps: `size_sweep_manifest.jsonl` is one line per leg
`hpc/batch_gpu/submit_size_sweep.sh` submitted (`job`, `mesh`, `path`, `n`, `node`, `submitted_utc`, `sha`),
appended at submit time and committed from the RAL worktree.

It is what lets `scripts/misc/likelihood_breakdown/size_sweep_table.py` tell a leg that ran out of memory
from a leg that was never submitted — the result JSONs themselves live under `results/breakdown/imaging/`.
