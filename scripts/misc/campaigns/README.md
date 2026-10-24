# campaigns

Work lists for the likelihood-lifecycle profiling campaigns, and the SLURM submits made from
them. One module per campaign; each holds the campaign's cells as tasks and renders the
submits under `hpc/` from those tasks, so the list the tests check and the jobs RAL runs are
the same list.

| Module | Campaign | Submits |
|--------|----------|---------|
| [`imaging_p2a.py`](imaging_p2a.py) | Phase 2a, `imaging/{mge, rectangular, delaunay, delaunay_nn}` at the standard size on 2026.10.23.1 (autolens_profiling#456) | `hpc/batch_cpu/submit_campaign_p2a_imaging_*`, `hpc/batch_gpu/submit_campaign_p2a_imaging_*` |

```bash
python scripts/misc/campaigns/imaging_p2a.py --list    # the 106 cells, script and flags
python scripts/misc/campaigns/imaging_p2a.py --write   # re-render the submits after an edit
python scripts/misc/campaigns/imaging_p2a.py --check   # exit 1 if a submit is out of date
pytest scripts/misc/campaigns/test -q
```

Never edit a generated submit by hand; edit the module and run `--write`.

## Phase 2a imaging

**Cells.** The 106 planned imaging cells (`dashboard/coverage.json`): 44 on `hpc_ral_cpu` and
62 on `hpc_a100`. Every pixelized cell runs with `--standard-size` (Delaunay 1250 Hilbert
vertices, plus the 30 zeroed edge points of the numba production construction; rectangular
35x35). Each row lands at its mirrored cell (`--output-path results/imaging/...`) with
`fit.png` and `parity.json` beside it. The numba leg of `delaunay_nn` runs the numba Delaunay
script with `--mesh-class delaunay_nn`.

**Submits.**

| Submit | Partition | Tasks |
|--------|-----------|-------|
| `batch_cpu/submit_campaign_p2a_imaging_ral_cpu_fp64` | `ral`, array | 44, one cell each |
| `batch_gpu/submit_campaign_p2a_imaging_a100_<instrument>_fp64` | `gpu`, `--gres=gpu:1` | one job per instrument: ao 14, euclid 14, hst 20, jwst 14 legs in turn |
| `batch_cpu/submit_campaign_p2a_imaging_batch0_ral_cpu_fp64` | `ral`, array | 14 (batch 0) |
| `batch_gpu/submit_campaign_p2a_imaging_batch0_a100_fp64` | `gpu`, `--gres=gpu:1`, array `%2` | 20 (batch 0) |

The 18 `jax_cpu_sparse_fp64@hpc_a100` cells run on the A100 node's CPU inside the GPU jobs.
No CPU-only job goes on the `gpu` partition.

Every submit loads the libraries from a private checkout at tag 2026.10.23.1
(`PYAUTO_PRIVATE_BASE`, default `/mnt/ral/jnightin/profiling/PyAuto_2026.10.23.1`) over the
shared venv (`PYAUTO_HPC_BASE`), refuses to run on any other tag or import origin, and checks
each row's `library_revisions` against that checkout. A missing dataset is simulated, and its
adapt image cached, once per instrument under a lock (`prepare_dataset`). Each leg's wall
clock is appended to `output/p2a_logs/walls.tsv`.

**Batch 0 (timing arm).** Run the two `batch0` submits first. They run the cells that bound
every other instrument: ao (by far the most masked pixels at the 3.5" mask: about 385k,
against 43k for jwst, 15k for hst and 4k for euclid) for mge, rectangular and delaunay, and
hst for delaunay_nn, which exists only there. Their rows go to the gitignored
`output/p2a_batch0/` tree, never `results/`. Their WALL-BASIS rows are a declared guess
(`source: unmeasured`, 3x floor).

**Setting batch 1's `--time`.** For each batch-1 submit, take the batch-0 walls in
`walls.tsv`:

- RAL array: `--time` is at least 1.25 x the slowest batch-0 CPU leg of the same script.
- A100 job: `--time` is at least 1.25 x the sum of its legs' walls, each leg taken from the
  batch-0 leg of the same script and setup.

Then change the generator's batch-1 WALL-BASIS rows to `source: measured-wall` (with `wall:`
and `ref:` the batch-0 job), set `WALL_BASIS_STATUS=measured`, run `--write`, and confirm
`python scripts/misc/wall/check_submits.py --check` passes. Until then the batch-1 submits refuse
to run (`WALL_BASIS_STATUS=provisional`; `P2A_ALLOW_PROVISIONAL_WALL=1` overrides it).

**Batch 1.** Submit the RAL array and the four A100 jobs from the private RAL checkout of
this repo. 13 of the target paths already hold older rows (hst, 1500 / 39x39). The run
overwrites them in that checkout, and PR B moves the old rows to their cell's `history/`.
