# campaigns

Work lists for the likelihood-lifecycle profiling campaigns, and the SLURM submits made from
them. One module per campaign; each holds the campaign's cells as tasks and renders the
submits under `hpc/` from those tasks, so the list the tests check and the jobs RAL runs are
the same list.

| Module | Campaign | Submits |
|--------|----------|---------|
| [`imaging_p2a.py`](imaging_p2a.py) | Phase 2a, `imaging/{mge, rectangular, delaunay, delaunay_nn}` at the standard size on 2026.10.23.1 (autolens_profiling#456) | `hpc/batch_cpu/submit_campaign_p2a_imaging_*`, `hpc/batch_gpu/submit_campaign_p2a_imaging_*` |
| [`peak_rss_backfill.py`](peak_rss_backfill.py) | Phase 2x backfill of 2a's peak memory and cores per call from sacct (autolens_profiling#462) | writes `catalogue/peak_rss_backfill.json`, no submits |

```bash
python scripts/misc/campaigns/imaging_p2a.py --list    # the 106 cells, script and flags
python scripts/misc/campaigns/imaging_p2a.py --write   # re-render the submits after an edit
python scripts/misc/campaigns/imaging_p2a.py --check   # exit 1 if a submit is out of date
pytest scripts/misc/campaigns/test -q
```

Never edit a generated submit by hand; edit the module and run `--write`.

## Phase 2a imaging

**Cells.** The 106 planned imaging cells (`dashboard/coverage.json`): 44 on `hpc_ral_cpu` and
62 on `hpc_a100`. Every pixelized cell runs with `--standard-size`: JAX and numba alike build
the numba production preset of the instrument from `_production_config.py` (jwst and ao
borrow the hst preset, recorded in `size_override`): Delaunay 1250 Hilbert vertices + 30
zeroed edge points (1280 built) or rectangular 35x35, the preset's MGE lens light, adapt-image
weights, regularization priors, over-sampling, settings and positions penalty, evaluated at
the preset's seeded iid draw 0. The data and mesh follow real production
(`production_standard_data`, `PRODUCTION_DATA_SOURCES` in `_production_config.py`): the
source adapt image is production's subtracted S/N map (data minus the truth lens light, over
the noise map, floored at 1 %; the dataset's truth tracer stands in for production's
previous-stage lens model, recorded as `production_data.lens_light_source: "truth"`), the over-sampling cut (sub-size 4 where S/N > 3, else 2) reads
the uncapped map, the S/N cap of 3 feeds only the mesh and regularization, and the
rectangular mesh is `RectangularBilinearAdaptImage` with free weight priors. `--standard-size`
refuses `--source-pixels`, `--rect-mesh`, `--over-sampling`, `--regularization` and `--variant`. So every setup of one likelihood and instrument evaluates the
same likelihood; the row records the preset as `configuration.production_preset`. Each row
gets `fit.png` and `parity.json` (the eager NumPy dense-mapping reference) beside it. The
numba leg of `delaunay_nn` runs the numba Delaunay script with `--mesh-class delaunay_nn`.

**Submits.**

| Submit | Partition | Tasks |
|--------|-----------|-------|
| `batch_cpu/submit_campaign_p2a_imaging_ral_cpu_<instrument>_fp64` | `ral`, array | one array per instrument: ao 10, euclid 10, hst 14, jwst 10 cells (44) |
| `batch_gpu/submit_campaign_p2a_imaging_a100_<instrument>_fp64` | `gpu`, `--gres=gpu:1` | one job per instrument: ao 14, euclid 14, hst 20, jwst 14 legs in turn |
| `batch_cpu/submit_campaign_p2a_imaging_batch0_ral_cpu_fp64` | `ral`, array | 14 (batch 0) |
| `batch_gpu/submit_campaign_p2a_imaging_batch0_a100_fp64` | `gpu`, `--gres=gpu:1`, array `%2` | 20 (batch 0) |

The 18 `jax_cpu_sparse_fp64@hpc_a100` cells run on the A100 node's CPU inside the GPU jobs.
No CPU-only job goes on the `gpu` partition.

Every submit loads the libraries from a private checkout at tag 2026.10.23.1
(`PYAUTO_PRIVATE_BASE`, default `/mnt/ral/jnightin/profiling/PyAuto_2026.10.23.1`) over the
shared venv (`PYAUTO_HPC_BASE`), refuses to run on any other tag or import origin, and checks
each row's `library_revisions` against that checkout. A missing dataset is simulated, and its
adapt image cached, once per instrument under a lock (`prepare_dataset`). Each leg appends a
`START` line and, when it finishes, a `WALL` line to `output/p2a_logs/walls.tsv`, so a leg
killed at `--time` shows as a `START` with no `WALL`.

**Memory.** `--mem` follows the dataset size (`mem_gb`): masked pixels at the 3.5" mask
(ao 384845, jwst 42760, hst 15393, euclid 3848) times 1340 dense-mapping columns, eight fp64
arrays of that size (the cell's mapping and blurred mapping matrices, the same two of the
parity reference, and working copies) plus 4 GB, doubled and rounded up: 128 GB for ao,
16 GB for the other RAL arrays, 32 GB minimum for the A100 jobs. That estimate sized batch 0;
batch 1 uses batch 0's `sacct --format=MaxRSS` instead (`mem_gb_measured`): the peak was 10.2 GiB
on hst (412877_5) and 35.1 GiB on ao (412877_1), jwst is interpolated linearly in masked pixels
and euclid takes the hst peak, and the request is the next 16/32/64/128 GB step at or above 1.5x
that peak, never below 32 GB (`BATCH1_MEM_FLOOR_GB`: the hst and euclid rectangular and delaunay
legs were never measured on their own instrument): ao 64 GB, every other submit 32 GB.

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

Done for 2026.10.23.1: the generator's `BATCH0_WALLS` holds every batch-0 leg (RAL 412877,
A100 412884, breakdown re-runs 412911 / 412915 after PR #458) as its `walls.tsv` WALL, its task's
sacct Elapsed, and the task; a batch-1 leg takes the larger of the two from its batch-0 leg of
the same script, setup and host class on the bounding instrument (`basis_leg`: ao, or hst for
delaunay_nn), so jwst, hst and euclid run under the ao upper bound. `--time` is 1.25x that plus
15 min of slack (`WALL_MIN_SLACK_S`: an A100 job that overruns loses every leg still to run),
rounded up to the next half hour: 1:00:00 for every RAL array, 3:00:00 for the ao, euclid and
jwst A100 jobs and 3:30:00 for hst (its six delaunay_nn legs on top).

The ao `jax_gpu_dense_fp64@hpc_a100 [breakdown]` legs of rectangular and delaunay ran out of
A100 memory in batches 0 and 1 and were first recorded as a known limitation
(`BATCH0_LIMITATIONS`). The cause was the breakdown holding duplicate device copies of the
384,753 x 1,280 fp64 mapping matrices; autolens_profiling#466 frees them and both legs ran
(RAL 413033), so `BATCH0_LIMITATIONS` is now empty. The rerun loaded the same private
2026.10.23.1 checkout as batch 1 (the same library SHAs), deliberately, so its rows are
like-for-like with the rest of 2a: they were **measured on 2026.10.23.1** and are published at
`release_pin` 2026.10.25.1 only by the declared carry-forward in
`catalogue/release_revisions.json` (`measured_release_pin` 2026.10.23.1), not by a measurement
on 2026.10.25.1.

**Batch 1.** Submit the four RAL arrays and the four A100 jobs from the private RAL checkout of
this repo. Rows go to the gitignored `output/p2a_batch1/results/...` scratch tree, never in
place: 13 of the target paths in `results/` already hold older rows (hst, 1500 / 39x39).

**PR B** copies only complete cells from `output/p2a_batch1/results/` into `results/`: the row
JSON exists and is the final write (no `"vmap": "PENDING ..."` early row), `fit.png` and
`parity.json` sit beside it, `parity_witness.py --check` passes on it, and its
`library_revisions` match the pinned checkout. Before copying, PR B moves each superseded row
(and its figure and witness) to the cell's `history/<release_pin>_<date>/`, then regenerates
the catalogue and coverage.

**Comparing CPU and GPU setups.** A per-call time is per device, not per node: numba runs one
serial likelihood per core under Nautilus multiprocessing, so never headline a CPU-vs-GPU
ratio from per-call latency. The catalogue's `projected_node_throughput` (autolens_profiling#462)
projects calls per second onto the declared node (`catalogue/reference_hosts.json`) the row was
measured on, never onto another machine: workers =
min(cores / `cpus_per_eval`, memory / `peak_rss_gb`), one per GPU on a GPU node, times the
declared parallel efficiency 1.0 (an upper bound, unmeasured until phase 4) over the per-call
time; Nautilus `n_batch` is not modelled. From 2a:

| cell | setup @ node | per call | workers (bound by) | projected calls/s per node |
|---|---|---|---|---|
| hst Delaunay | `numba_cpu_sparse_fp64` @ ral node (236 cores, 921 GB) | 951.7 ms | 236 (cores; 0.85 GB each) | ≈248 |
| hst Delaunay | `jax_gpu_dense_fp64` @ gpu node (4×A100) | 35.05 ms | 4 (GPUs) | ≈114 |
| ao Delaunay | `numba_cpu_sparse_fp64` @ ral node | 8,671 ms | 40 (memory; 23.0 GB each) | ≈4.6 |
| ao Delaunay | `jax_gpu_dense_fp64` @ gpu node | 366.93 ms | 4 (GPUs) | ≈10.9 |

The full table and caveats are on the
[2a campaign page](../../../wiki/campaigns/imaging/likelihood_lifecycle_p2a.md#comparing-cpu-and-gpu-setups).

**Peak memory backfill.** Rows written before #462 have no `resources` block.
`peak_rss_backfill.py --sacct <dump> --write` maps each committed batch-1 row to its task
(`batch1_ral_cpu(instrument)[i]` for RAL array task `i`; every leg of an A100 job), checked
against `walls.tsv`, and records the `.batch` step's MaxRSS. New rows record their own
in-process peak (`_profile_cli.resources_dict`), so later campaigns need no backfill.

**For PR B's campaign wiki page.**

- The euclid positions penalty is part of every euclid runtime row (JAX and numba time
  `log_likelihood_function`), but the JAX breakdowns decompose `FitImaging`, which has no
  positions term: a euclid JAX breakdown does not time the penalty.
- Without `--standard-size`, the numba production presets still build their data the historic
  way: the adapt image is the noise-free lensed-source flux, capped at 3 before it drives the
  over-sampling cut (so on hst, jwst and ao the cut never fires and every pixel gets sub-size 2),
  and the rectangular presets use `RectangularBilinearAdaptDensity`. Production does neither;
  only the standard-size cells match production.
