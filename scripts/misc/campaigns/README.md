# campaigns

Work lists for the likelihood-lifecycle profiling campaigns, and the SLURM submits made from
them. One module per campaign; each holds the campaign's cells as tasks and renders the
submits under `hpc/` from those tasks, so the list the tests check and the jobs RAL runs are
the same list.

| Module | Campaign | Submits |
|--------|----------|---------|
| [`imaging_p2a.py`](imaging_p2a.py) | Phase 2a, `imaging/{mge, rectangular, delaunay, delaunay_nn}` at the standard size on 2026.10.23.1 (autolens_profiling#456) | `hpc/batch_cpu/submit_campaign_p2a_imaging_*`, `hpc/batch_gpu/submit_campaign_p2a_imaging_*` |
| [`imaging_p2a.py`](imaging_p2a.py) (`jax_rerun_tasks`) | Phase 3 day-zero JAX re-measure of the 2a cells on 2026.10.25.1 under jax 0.11.2 (human 2026-10-30, Q7) | `hpc/batch_cpu/submit_campaign_p2a_jaxrerun_ral_cpu_*_fp64`, `hpc/batch_gpu/submit_campaign_p2a_jaxrerun_a100_*_fp64` |
| [`imaging_p2a.py`](imaging_p2a.py) (`rect28_tasks`) | Phase 2a size variant: hst rectangular at 28x28 = 784 source pixels on 2026.10.25.1 (autolens_profiling#468) | `hpc/batch_cpu/submit_campaign_p2a_rect28_ral_cpu_fp64`, `hpc/batch_gpu/submit_campaign_p2a_rect28_a100_fp64` |
| [`interferometer_p2b.py`](interferometer_p2b.py) | Phase 2b, `interferometer/{mge, rectangular, delaunay}` at the standard size on 2026.10.25.1 (autolens_profiling#469), plus the dense parity references and the matrix-free experiment | `hpc/batch_cpu/submit_campaign_p2b_*`, `hpc/batch_gpu/submit_campaign_p2b_*` |
| [`point_source_p2c.py`](point_source_p2c.py) | Phase 2c, `point_source_image/image_plane` and `point_source_source/source_plane` (instrument `simple`) on 2026.10.25.1 (autolens_profiling#472), plus the free-centre (plain) runtime experiments | `hpc/batch_cpu/submit_campaign_p2c_point_source_ral_cpu_fp64`, `hpc/batch_gpu/submit_campaign_p2c_point_source_a100_fp64` |
| [`multi_dataset_datacube_p2f.py`](multi_dataset_datacube_p2f.py) | Phase 2f, `multi_dataset/delaunay` and `datacube/delaunay` at the standard size on 2026.10.25.1 (autolens_profiling#475); batch 1 run 2026-10-30, its 32 rows committed | `hpc/batch_cpu/submit_campaign_p2f_*`, `hpc/batch_gpu/submit_campaign_p2f_*` |
| [`weak_p2g.py`](weak_p2g.py) | Phase 2g, `weak/isothermal/simple` (new weak cells) on 2026.10.25.1 (autolens_profiling#476); batch 0 run 2026-10-29; its rows are the campaign's rows (D9), committed under `results/weak/` | `hpc/batch_cpu/submit_campaign_p2g_*`, `hpc/batch_gpu/submit_campaign_p2g_*` |
| [`multi_galaxy_group_p2d.py`](multi_galaxy_group_p2d.py) | Phase 2d, `{multi_galaxy,group}/{mge, rectangular, delaunay}` at the standard size on 2026.10.25.1 (autolens_profiling#473); batch 1 run 2026-10-30, its 48 rows committed | `hpc/batch_cpu/submit_campaign_p2d_*`, `hpc/batch_gpu/submit_campaign_p2d_*` |
| [`cluster_p2e.py`](cluster_p2e.py) | Phase 2e, `cluster/{image_plane, source_plane}` on 2026.10.25.1 (autolens_profiling#474); batch 1 run 2026-10-30, its 10 rows committed | `hpc/batch_cpu/submit_campaign_p2e_*`, `hpc/batch_gpu/submit_campaign_p2e_*` |
| [`_common.py`](_common.py) | The submit machinery the generators share (preamble, guards, WALL-BASIS rows, `#SBATCH` block) | none of its own |
| [`peak_rss_backfill.py`](peak_rss_backfill.py) | Phase 2x backfill of 2a's peak memory and cores per call from sacct (autolens_profiling#462) | writes `catalogue/peak_rss_backfill.json`, no submits |

```bash
python scripts/misc/campaigns/imaging_p2a.py --list    # the 106 cells + 8 28x28 legs, script and flags
python scripts/misc/campaigns/imaging_p2a.py --write   # re-render the submits after an edit
python scripts/misc/campaigns/imaging_p2a.py --check   # exit 1 if a submit is out of date
python scripts/misc/campaigns/interferometer_p2b.py --list    # the 60 cells, 6 references, 7 matrix-free legs
python scripts/misc/campaigns/interferometer_p2b.py --write   # re-render the 2b submits
python scripts/misc/campaigns/interferometer_p2b.py --check
python scripts/misc/campaigns/point_source_p2c.py --list      # the 10 cells and 5 plain-runtime experiments
python scripts/misc/campaigns/point_source_p2c.py --write     # re-render the 2c submits
python scripts/misc/campaigns/point_source_p2c.py --check
python scripts/misc/campaigns/multi_dataset_datacube_p2f.py --list    # the 32 cells (D1) and 2 references
python scripts/misc/campaigns/multi_dataset_datacube_p2f.py --write   # re-render the 2f submits
python scripts/misc/campaigns/multi_dataset_datacube_p2f.py --check
python scripts/misc/campaigns/weak_p2g.py --list    # the 6 planned weak cells, all runnable
python scripts/misc/campaigns/weak_p2g.py --write
python scripts/misc/campaigns/weak_p2g.py --check
python scripts/misc/campaigns/multi_galaxy_group_p2d.py --list   # the 48 planned 2d cells
python scripts/misc/campaigns/multi_galaxy_group_p2d.py --write  # re-render the 2d submits
python scripts/misc/campaigns/multi_galaxy_group_p2d.py --check
python scripts/misc/campaigns/cluster_p2e.py --list    # the 10 planned cluster cells, all runnable
python scripts/misc/campaigns/cluster_p2e.py --write   # re-render the 2e submits
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

## Phase 3 day-zero JAX re-measure

**Why.** Day zero is one environment (human 2026-10-30, Q7 of the phase-3 designation
proposals): the 2a/2b/2c JAX rows ran on jax/jaxlib 0.10.2, which tag 2026.10.25.1's PyAutoNerves
excludes, while 2d-2g ran on 0.11.2 with the eigen flag (D12). Every 2a/2b/2c JAX row is
re-measured on 0.11.2; numba rows are carried. The tag stays 2026.10.25.1 (Q8).

**Legs.** 2a: `jax_rerun_tasks(instrument, host_class)` (the JAX legs of each batch-1 submit,
26 RAL CPU + 62 A100) on its own pin (2026.10.25.1, not 2a's 2026.10.23.1) into the gitignored
`output/p3_jax_rerun/` tree, under the batch-1 measured walls. 2b and 2c: their existing
batch-1 A100 submits (every A100 leg is JAX, already on 2026.10.25.1); their RAL CPU JAX legs
come from the quiet reruns (RAL 413319-413323, already on 0.11.2 with the flag). Rows are copied
into `results/` by the phase-3 rows PR, superseded rows moved to `history/` first.

## Rectangular 28x28 size variant

**Why.** The SLaM inference runs fit rectangular sources on a 28x28 mesh (784 source pixels),
while the profiling standard is 35x35 (1225, human size rule 2026-10-23), so every
autolens_inference wall-time row paired with a profiling runtime row of another size
(autolens_inference#47, `size_match: false`). This batch measures hst rectangular at 784 beside
the 35x35 standard, which stays the coverage cell (human 2026-10-26, autolens_profiling#468).
Delaunay needs nothing: its runs use 1250, the standard.

**Cells.** Eight legs, `rect28_tasks()`: runtime and breakdown of `numba_cpu_sparse_fp64@hpc_ral_cpu`,
`jax_cpu_sparse_fp64@hpc_a100`, `jax_gpu_sparse_fp64@hpc_a100` and `jax_gpu_dense_fp64@hpc_a100`,
the setups the inference runs record. Each leg runs the 2a standard cell's script and flags plus
`--source-pixels 784`: `--standard-size --source-pixels 784` builds the same standard-size
production cell (the hst preset, production data, MGE, regularization, settings, positions
penalty) with the mesh resized to 28x28 instead of 35x35. `--standard-size` accepts
`--source-pixels` only for a count declared in
[`catalogue/size_variants.json`](../../../catalogue/size_variants.json) (rectangular 784);
any other count is still refused. `--source-pixels 784` alone would not do: the JAX cells would
build their legacy fiducial cell at 28x28 and the numba cells ignore it.

**Submits.**

| Submit | Partition | Tasks |
|--------|-----------|-------|
| `batch_cpu/submit_campaign_p2a_rect28_ral_cpu_fp64` | `ral`, array `0-1%16` | 2 numba legs, `--time=1:00:00`, `--mem=32gb` |
| `batch_gpu/submit_campaign_p2a_rect28_a100_fp64` | `gpu`, `--gres=gpu:1` | 6 legs in turn, `--time=1:30:00`, `--mem=32gb` |

The two `jax_cpu_sparse_fp64@hpc_a100` legs run on the A100 node's CPU inside the GPU job
(`JAX_PLATFORMS=cpu`, `NPROC=8`). Both submits load the private checkout at tag
**2026.10.25.1** (`PYAUTO_PRIVATE_BASE`, default
`/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1`, `P2A_PIN=2026.10.25.1`) with the same
import-origin, tag and `library_revisions` guards as 2a, so the rows publish `release_pin`
2026.10.25.1 by exact SHA match (`catalogue/release_revisions.json`), with no carry-forward.

**Walls and memory.** Nothing ran at 28x28 yet, and batch 0 measured no hst rectangular leg, so
each leg takes its 2a standard leg's batch-0 wall on ao (`basis_leg`) and `--mem` takes the hst
MaxRSS (`mem_gb_measured`, 32 GB floor). The 784-pixel mesh is smaller than 1225 and hst has 4%
of ao's masked pixels, so both are upper bounds.

**Rows.** They go to the gitignored `output/p2a_rect28/results/...` tree. Copy complete cells
(row, `fit.png`, `parity.json`, revisions checked) into the cell's `source_pixels_784/` folder,
e.g. `results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/source_pixels_784/runtime.json`.
The catalogue reads that folder as the declared size variant: its rows are keyed on their own,
`regime: source_pixels_784`, `current` beside the standard row with a `size_variant` limitation
and no `size_fallback` note. Coverage and the trend dashboard stay standard-only.

**Validation:** PyAutoPulse#80 adds the regime to the v3 one-`current` key, so the 784 rows
pass `build_catalogue.py --check --validate-with ../PyAutoPulse` beside the standard row.

**Ran:** RAL 413044 (CPU array) and 413046 (A100), 8/8 legs exit 0, every leg's
`library_revisions` at tag 2026.10.25.1, so the rows publish `release_pin` 2026.10.25.1 by exact
SHA match. The rows still stamp `autolens_version` 2026.8.17.1 (the source checkout's stale
`__version__`); the revisions are the provenance.

## Phase 2b interferometer

**Cells.** The 60 planned interferometer cells (`dashboard/coverage.json`): 38 on `hpc_ral_cpu`
and 22 on `hpc_a100`. The human's rulings (autolens_profiling#469) set the list:

- R1(a): dense pixelized NUFFT (`jax_*_dense_fp64_nufft`) is planned at sma only. Above sma
  the library's one-shot dense transform does not fit in memory, so 84 cells became 60.
- R2: no DFT cells. R9: the registry's instruments, so MGE runs at sma only.
- Rectangular and Delaunay: `numba_cpu_sparse_fp64_wtilde` (optimal) and
  `jax_cpu_sparse_fp64_wtilde` on every instrument, `jax_cpu_dense_fp64_nufft` at sma on `ral`;
  `jax_gpu_sparse_fp64_wtilde` (optimal) everywhere and `jax_gpu_dense_fp64_nufft` at sma on the
  A100. MGE: `jax_cpu_dense_fp64_nufft` and `jax_gpu_dense_fp64_nufft` at sma.

Every pixelized leg runs with `--standard-size` (R3): the interferometer production cell from
`_production_config.interferometer_standard_preset`. That is Delaunay 1250 Hilbert vertices
plus 30 zeroed edge points with `AdaptSplit`, or rectangular 35x35
`RectangularBilinearAdaptImage` with `Adapt`. The adapt image is the truth lensed source, not
capped. There is no lens light. MGE-20 is unchanged. Each setup of one likelihood and
instrument evaluates the same likelihood at the preset's seeded iid draw 0.

The flags each leg gets:

- JAX runtime: `--sparse` for the W~ setups, nothing for the dense ones.
- JAX breakdown: the W~ harness is the default; `--dense` writes the dense row.
- numba runtime and breakdown: `--n-instances` 20 / 20 / 10 / 5 on sma / alma / alma_high /
  jvla (R4). The headline is the route the packaged cost model picks; the forced routes are
  sub-rows.

**Submits.**

| Submit | Partition | Tasks | `--time` | `--mem` |
|--------|-----------|-------|----------|---------|
| `batch_cpu/submit_campaign_p2b_batch0_ral_cpu_fp64` | `ral`, array `%8` | 14 (batch 0) | 12:00:00 | 128 GB |
| `batch_gpu/submit_campaign_p2b_batch0_a100_fp64` | `gpu`, `--gres=gpu:1`, array `%2` | 10 (batch 0) | 6:00:00 | 64 GB |
| `batch_cpu/submit_campaign_p2b_batch0_references_ral_cpu_fp64` | `ral`, array | 2 (optional, batch 0) | 24:00:00 | 64 GB |
| `batch_cpu/submit_campaign_p2b_interferometer_ral_cpu_<instrument>_fp64` | `ral`, array `%16` | sma 14, alma 8, alma_high 8, jvla 8 | 2:30:00 | 64 GB |
| `batch_gpu/submit_campaign_p2b_interferometer_a100_<instrument>_fp64` | `gpu`, `--gres=gpu:1` | one job per instrument: sma 10, others 4 legs in turn | 1:30:00 | 64 GB |
| `batch_cpu/submit_campaign_p2b_references_ral_cpu_fp64` | `ral`, array | 6 (alma, alma_high, jvla x rectangular, delaunay) | 4:00:00 | 64 GB |
| `batch_cpu/submit_campaign_p2b_matrixfree_ral_cpu_fp64` | `ral`, array | 7 (experiment, not batch 0) | 12:00:00 | 128 GB |

There are no `jax_cpu@hpc_a100` legs in 2b, so every A100 leg is a GPU leg. No CPU-only job
goes on the `gpu` partition.

**Pin and guards.** Every campaign submit loads the libraries from a private checkout at tag
2026.10.25.1 (R5, changed by the human 2026-10-26; `PYAUTO_PRIVATE_BASE`, default
`/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1`) over the shared venv (`PYAUTO_HPC_BASE`;
its finufft 2.5.1 serves the NumPy NUFFT). The guards are 2a's: the job refuses any other tag
or import origin, and checks each row's `library_revisions` against the checkout. A missing
dataset is simulated under a lock (`prepare_dataset`), which also writes `lensed_source.fits`
and the NumPy W~ preload cache, so array tasks never race to build them. Logs and the
`walls.tsv` START / WALL lines go to `output/p2b_logs/`.

**Memory.** The `--mem` values are declared estimates from the dataset sizes (2b survey A8),
not measurements. jvla's mapping matrix alone is 384,852 x 1,280 x 8 B = 3.9 GB, and the JAX
CPU W~ compile temporaries there have never been measured, so jvla CPU asks 128 GB. The batch-0
RAL array covers sma and jvla in one array, so it takes jvla's 128 GB for every task. That
wastes memory on the sma tasks, but a `ral` node has 921 GB and one array is simpler to submit
and to read back. Batch 1 replaces the estimates with batch 0's `sacct --format=MaxRSS`
(`mem_gb_measured`: 1.5 x the peak of the bounding instrument, at least 32 GB).

**Batch 0 (timing arm, R8).** Run the batch-0 submits first. They run the legs that bound the
others: jvla for every W~ and numba setup (the largest mask and visibility count, never
measured on CPU), and sma for the dense and MGE setups, which exist only there. Rows go to the
gitignored `output/p2b_batch0/`. The WALL-BASIS rows are declared guesses (`source:
unmeasured`, 3x floor), and batch 0 carries no provisional guard, so it runs as generated. The
two jvla reference tasks are optional; running them in batch 0 measures their wall.

**Setting batch 1's `--time`.** Fill `BATCH0_WALLS` (each batch-0 leg's `walls.tsv` WALL, its
sacct Elapsed and the task) and `BATCH0_PEAK_RSS_KIB` / `REFERENCE_PEAK_RSS_KIB` in
`interferometer_p2b.py`, set `WALL_BASIS_STATUS = "measured"`, run `--write`, and confirm
`python scripts/misc/wall/check_submits.py --check` passes. Each batch-1 leg then takes the wall
of its batch-0 leg on the bounding instrument (`basis_leg`), with 2a's rule: 1.25x, plus 15 min,
rounded up to the next half hour. Until then the batch-1 submits refuse to run
(`P2B_ALLOW_PROVISIONAL_WALL=1` overrides).

Done for 2026.10.25.1 (RAL 413115, references 413116, A100 413117; all 26 legs exit 0, every
jvla parity witness `match`): `BATCH0_WALLS` holds every batch-0 leg. Each batch-1 leg takes the
larger of WALL and Elapsed of its batch-0 leg (jvla for the sparse, numba and reference legs, so
alma and alma_high run under the jvla upper bound: about 6x and 25x fewer masked pixels; sma for
the dense and MGE legs). The slowest leg is the rectangular jvla numba breakdown (6461 s), so
every RAL array asks 2:30:00; the A100 jobs sum their legs (sma 2728 s, the others 2278 s) and
ask 1:30:00; the references take the delaunay jvla reference (10727 s) and ask 4:00:00. `--mem`
is 64 GB everywhere: the cell peak was 30.8 GiB on sma (413115_5, the delaunay JAX CPU dense
runtime) and 21.4 GiB on jvla (413115_1), alma and alma_high take jvla's, the references
peaked at 30.5 GiB (413116_1), and 1.5x each lands on the 64 GB step. The four ~25 s MGE legs
report no MaxRSS (they end before sacct samples them); their memory is bounded by the sma
pixelized legs in the same job.

**Parity references (R6, R7).** Every pixelized cell writes `parity.json` against the
visibility-chunked eager NumPy dense log evidence at the same instance. At sma a cell computes
it in-process. Above sma it is one RAL task per likelihood and instrument
(`scripts/interferometer/pixelized/parity_reference.py`), cached in `output/p2b_references/`.
A cell that finishes before its reference writes a `not_assessed` witness and keeps its value
as `parity_candidate`; the reference task's `--results-root` pass then rewrites the witness.
To finish later by hand, run the script with `--finalise-only`. The tolerance is 1e-8 x
|reference| in fp64 and 1e-3 in mixed precision.

**Matrix-free experiment.** `submit_campaign_p2b_matrixfree_ral_cpu_fp64` runs the NumPy CPU
matrix-free route (`Settings(interferometer_route="matrix_free")`, PyAutoArray#658) on a
`RectangularUniform` mesh with `MaternSparse` regularization. It covers 35x35 on all four
instruments and an alma mesh scan at 64x64, 128x128 and 192x192, and records the log-evidence
agreement with the dense reference. Rows are experiments only, never current: 35x35 under the
cell's `experiments/numpy_cpu_matrixfree/`, the scan under `scans/numpy_cpu_matrixfree_<N>x<N>/`,
both in `numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/` (the setup id grammar has no `numpy_`
backend), in the gitignored `output/p2b_matrixfree/`. It is not part of batch 0.

#658 merged after 2026.10.25.1, so this submit loads its own private checkout of library main
(`MF_PRIVATE_BASE`, default `/mnt/ral/jnightin/profiling/PyAuto_main_7eba0c07`) and refuses to
run unless all five HEADs equal `MF_LIBRARY_SHAS` (origin/main on 2026-10-27). It is labelled
unreleased. Open question for the human: PyAutoArray `7eba0c07` is also tag 2026.10.26.1, which
is on PyPI, so this experiment could instead pin that release.

Known gap (`BATCH1_LIMITATIONS`, human ruling 2026-10-29, "go ahead with cell missing"): the alma
192x192 scan leg (RAL 413191_6) hit TIMEOUT at 12:00:29. Its 20 matrix-free instances finished
(about 12-15 s each); the dense_fft comparison arm at 36,864 source pixels did not, so it wrote no
row. It is recorded as missing and not resubmitted. The other six matrix-free legs landed.

**PR B** copies only complete cells from `output/p2b_batch1/results/` into `results/`, as in 2a:
the final row, `fit.png`, a `parity.json` that passes `parity_witness.py --check` (not
`not_assessed`), and `library_revisions` that match the pinned checkout. It moves the 20
`size_fallback` rows and the stale MGE row to `history/<release_pin>_<date>/` first, then
regenerates the catalogue and coverage. Done for batch 1: all 60 cells (30 witnesses, all `match`) and 6
of the 7 matrix-free legs; the 21 superseded rows are in `history/`. The campaign page is
[`likelihood_lifecycle_p2b.md`](../../../wiki/campaigns/interferometer/likelihood_lifecycle_p2b.md).

## Phase 2d multi_galaxy + group

**Status.** Batch 0 ran 2026-10-29 (RAL 413272, A100 413294); batch 1 awaits the human's go
(autolens_profiling#473). The cell design is
decisions D1-D12 on the campaign page (D12, decided by the human 2026-10-29: the JAX legs append `--xla_cpu_multi_thread_eigen=false`, an XLA:CPU deadlock workaround),
[`wiki/campaigns/multi_galaxy/likelihood_lifecycle_p2d.md`](../../../wiki/campaigns/multi_galaxy/likelihood_lifecycle_p2d.md).
Each batch needs the human's go.

**Cells.** 48 planned cells (`catalogue/registry.json`, §3.3 "as imaging"): `multi_galaxy` at
hst and `group` at euclid, each `mge` (`jax_cpu_dense_fp64`@`hpc_ral_cpu`,
`jax_gpu_dense_fp64`@`hpc_a100`) and `rectangular` / `delaunay` (`numba_cpu_sparse_fp64` and
`jax_cpu_sparse_fp64`@`hpc_ral_cpu`; `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64` and
`jax_cpu_sparse_fp64`@`hpc_a100`), runtime and breakdown: 20 RAL CPU, 28 A100. The scripts are
new (`scripts/{multi_galaxy,group}/<variant>/likelihood_{runtime,breakdown}[_numba].py`), thin
wrappers over one driver, `scripts/misc/likelihood_breakdown/galaxy_group_standard.py`, on one
cell builder, `scripts/misc/cell_builders/galaxy_group.py` (dataset simulation, masking, the
workspace modeling model, the standard-size pixelization). JAX legs take `--sparse` for the
sparse setups; nothing else but `--config-name`.

**Submits.**

| Submit | Partition | Tasks | `--time` | `--mem` |
|--------|-----------|-------|----------|---------|
| `batch_cpu/submit_campaign_p2d_batch0_ral_cpu_fp64` | `ral`, array `%8` | 10 (batch 0, every group RAL leg) | 4:00:00 | 32 GB |
| `batch_gpu/submit_campaign_p2d_batch0_a100_fp64` | `gpu`, `--gres=gpu:1`, array `%2` | 14 (batch 0, every group A100 leg) | 4:00:00 | 32 GB |
| `batch_cpu/submit_campaign_p2d_<dataset_type>_ral_cpu_<instrument>_fp64` | `ral`, array `%16` | 10 each | 0:30:00 | 16 GB |
| `batch_gpu/submit_campaign_p2d_<dataset_type>_a100_<instrument>_fp64` | `gpu`, `--gres=gpu:1` | one job each, 14 legs in turn | 1:00:00 | 16 GB |

The `jax_cpu_sparse_fp64@hpc_a100` legs run on the GPU node's CPU inside the GPU job, as in 2a.
Batch 0 is the timing arm (D9): group bounds multi_galaxy (1.6x the masked pixels, two dPIE
extra galaxies), so its legs set batch 1's walls. Its rows declare `source: unmeasured` with a
3600 s guess (3x floor). Batch 1 renders `WALL_BASIS_STATUS=provisional` and refuses to run
(`P2D_ALLOW_PROVISIONAL_WALL=1` overrides) until `BATCH0_WALLS` holds batch 0's walls and the
generator sets `WALL_BASIS_STATUS = "measured"`: then `--time` is 1.25x the bounding group leg (an
A100 job: the sum of its legs) plus 15 min, rounded up to 30 min, and `--mem` the next step at
or above 1.5x batch 0's MaxRSS, as 2a and 2b. Pin, private checkout and guards are 2b's
(`P2D_PIN=2026.10.25.1`); logs and `walls.tsv` go to `output/p2d_logs/`. A missing dataset is
simulated under a lock (`prepare_dataset`, by instrument: hst -> multi_galaxy, euclid -> group).

Done 2026-10-30: `BATCH0_WALLS` and `BATCH0_PEAK_RSS_KIB` hold every batch-0 leg (all exit 0,
every witness `match`; peak 3.0 GiB) and `WALL_BASIS_STATUS = "measured"`.

**Local check.** Every cell script takes `--fast` (2 repeats, a 5-call median, vmap batch 2, a
4-instance stream) and, without `--output-path`, writes into the gitignored
`output/p2d_local/results/...`, never `results/`.

## Phase 2g weak lensing

**Cells.** Weak lensing had no profiling script and no planned cell. 2g adds the registry cell
`weak/isothermal/simple`; the §3.3 weak row then plans 6 cells, runtime and breakdown:
`jax_cpu_dense_fp64` (provisional optimal) on both reference hosts and `jax_gpu_dense_fp64`
(secondary) on `hpc_a100` only (the planner plans a GPU setup only on a GPU host,
autolens_profiling#481, which settles decision D5). The manifest has 6 legs: 2 on `ral`, 4 in one
A100 job. The
`jax_cpu_dense_fp64@hpc_a100` legs run on the A100 node's CPU (`JAX_PLATFORMS=cpu`), before the
GPU legs in the same job. Every leg evaluates the standard weak cell
(`scripts/misc/likelihood_breakdown/weak_standard.py`): the `simple` catalogue of
`scripts/misc/simulators/weak.py` (1500 galaxies in the 50"-200" annulus, sigma 0.25, plain
shear), fitted with a free elliptical isothermal lens at the simulator truth. The decisions
behind this design are on the campaign page
([`wiki/campaigns/weak/likelihood_lifecycle_p2g.md`](../../../wiki/campaigns/weak/likelihood_lifecycle_p2g.md)).

**Submits.**

| Submit | Partition | Tasks | `--time` | `--mem` |
|--------|-----------|-------|----------|---------|
| `batch_cpu/submit_campaign_p2g_batch0_ral_cpu_fp64` | `ral`, array `%8` | 2 (batch 0) | 1:30:00 | 16 GB |
| `batch_gpu/submit_campaign_p2g_batch0_a100_fp64` | `gpu`, `--gres=gpu:1` | 4 legs in turn (batch 0) | 1:30:00 | 32 GB |
| `batch_cpu/submit_campaign_p2g_weak_ral_cpu_simple_fp64` | `ral`, array `%8` | 2 (batch 1) | 0:30:00 | 16 GB |
| `batch_gpu/submit_campaign_p2g_weak_a100_simple_fp64` | `gpu`, `--gres=gpu:1` | 4 legs in turn (batch 1) | 0:30:00 | 32 GB |

The pin, private checkout, import and revision guards, `walls.tsv` lines and dataset lock are
2b's (`_common.py`); the catalogue is simulated under `output/p2g_dataset_<instrument>.lock`.
Logs go to `output/p2g_logs/`, rows to `output/p2g_batch{0,1}/`.

**Batches.** Batch 0 runs every runnable cell once with declared wall guesses (`source:
unmeasured`, 1200 s per row, 3x floor; the laptop ran each leg in under 50 s). Batch 1 is the
same six legs and refuses to run while `WALL_BASIS_STATUS = "provisional"`
(`P2G_ALLOW_PROVISIONAL_WALL=1` overrides). After batch 0, fill `BATCH0_WALLS`, set
`WALL_BASIS_STATUS = "measured"` and `--write`, as in 2b. Batch 0's rows are at the pinned tag
from the private checkout at the standard size, so decision D9 proposes accepting them as the
campaign's rows and skipping batch 1 (adopted 2026-10-29). Batch 0 ran 2026-10-29 (RAL 413293,
A100 job 413297; every leg exit 0, every witness `match`); its walls are in `BATCH0_WALLS` as the
basis of any rerun.

## Phase 2f multi_dataset + datacube

**Status.** Set up for the human's review (2026-10-28); no batch has a go. The decisions D1-D9
are on [the datacube campaign page](../../../wiki/campaigns/datacube/likelihood_lifecycle_p2f.md);
the multi_dataset half has [its own page](../../../wiki/campaigns/multi_dataset/likelihood_lifecycle_p2f.md).

**Cells.** 32 planned cells under D1 (`dashboard/coverage.json`): 18 on `hpc_ral_cpu` and 14 on
`hpc_a100`.

- multi_dataset/delaunay at hst (10): `numba_cpu_sparse_fp64`, `jax_cpu_sparse_fp64` on `ral`;
  `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64` and `jax_cpu_sparse_fp64` (on the GPU node's CPU,
  inside the GPU job) on the A100. Every leg passes `--n-datasets 2` (D6) and `--sparse` on the
  W~ setups.
- datacube/delaunay at sma, alma, alma_high (22): `numba_cpu_sparse_fp64_wtilde` and
  `jax_cpu_sparse_fp64_wtilde` everywhere, `jax_cpu_dense_fp64_nufft` at sma (D1) on `ral`;
  `jax_gpu_sparse_fp64_wtilde` everywhere and `jax_gpu_dense_fp64_nufft` at sma on the A100.
  Every leg passes `--n-channels 34` (D2); the JAX runtime takes `--sparse` on W~, the JAX
  breakdown `--dense` on the dense setups, numba `--n-instances` 20 / 20 / 10 (2b's R4).

Every leg runs with `--standard-size`: the drivers
`scripts/misc/likelihood_breakdown/{datacube,multi_dataset}_standard.py` build the cube from 2b's
interferometer standard cell and the multi-dataset fit from 2a's hst imaging cell, as
`FactorGraphModel` sums (the production configurations), and write `resources`, `fit.png` and
`parity.json`.

**Submits.**

| Submit | Partition | Tasks | `--time` | `--mem` |
|--------|-----------|-------|----------|---------|
| `batch_cpu/submit_campaign_p2f_batch0_ral_cpu_fp64` | `ral`, array `%8` | 10 (batch 0) | 12:00:00 | 128 GB |
| `batch_gpu/submit_campaign_p2f_batch0_a100_fp64` | `gpu`, `--gres=gpu:1`, array `%2` | 10 (batch 0) | 6:00:00 | 64 GB |
| `batch_cpu/submit_campaign_p2f_batch0_references_ral_cpu_fp64` | `ral`, array | 1 (optional, batch 0) | 12:00:00 | 64 GB |
| `batch_cpu/submit_campaign_p2f_multi_dataset_ral_cpu_hst_fp64` | `ral`, array `%16` | 4 | 0:30:00 | 32 GB |
| `batch_gpu/submit_campaign_p2f_multi_dataset_a100_hst_fp64` | `gpu`, `--gres=gpu:1` | 6 legs in turn | 1:00:00 | 32 GB |
| `batch_cpu/submit_campaign_p2f_datacube_ral_cpu_<instrument>_fp64` | `ral`, array `%16` | sma 6, alma 4, alma_high 4 | 1:00:00 | sma 64, alma 64, alma_high 32 GB |
| `batch_gpu/submit_campaign_p2f_datacube_a100_<instrument>_fp64` | `gpu`, `--gres=gpu:1` | sma 4, alma 2, alma_high 2 legs in turn | sma 1:00:00, alma/alma_high 0:30:00 | sma 64, alma 64, alma_high 32 GB |
| `batch_cpu/submit_campaign_p2f_references_ral_cpu_fp64` | `ral`, array | 2 (alma, alma_high) | 1:30:00 | 64 GB |

**Pin and guards.** 2b's: tag 2026.10.25.1 from `PYAUTO_PRIVATE_BASE` (default
`/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1`), the import-origin and tag guard, the
per-leg revision check, `walls.tsv` START / WALL lines in `output/p2f_logs/`. `prepare_dataset`
dispatches to 2a's imaging preflight for hst and 2b's interferometer preflight (dataset,
`lensed_source.fits`, NumPy W~ preload) for the datacube instruments, under their locks.

**Batch 0 (timing arm, D9).** Run the batch-0 submits first, after the human's go. They run the
legs that bound the others: alma_high for every datacube W~ and numba setup, sma for its dense
setups, and all 10 multi_dataset legs (hst only). Rows go to the gitignored `output/p2f_batch0/`.
The WALL-BASIS rows are declared guesses (`source: unmeasured`, 3x floor) and the `--mem`
values estimates from the dataset sizes.

**Setting batch 1's `--time`.** As 2b: fill `BATCH0_WALLS` and `BATCH0_PEAK_RSS_KIB` in
`multi_dataset_datacube_p2f.py`, set `WALL_BASIS_STATUS = "measured"`, run `--write` and
`python scripts/misc/wall/check_submits.py --check`. Each batch-1 leg then takes the wall of its
batch-0 leg on the bounding instrument (`basis_leg`): 1.25x, plus 15 min, rounded up to the next
half hour. Done 2026-10-30 from batch 0 (RAL 413274, A100 413296, references 413292; every leg
exit 0, every witness `match`): batch 1 runs 0:30:00-1:00:00 (references 1:30:00) at 32-64 GB.

**Parity references (D3, D8).** A datacube cell's reference is derived from 2b's chunked dense
reference of (delaunay, instrument) at the same instance: in-process at sma, from
`output/p2b_references/` above sma when 2b's reference task has written it in the submitting
checkout, else `not_assessed` until `scripts/datacube/delaunay/parity_reference.py`
(`--results-root output/p2f_batch1`, or `--finalise-only`) rewrites the witness. A
multi_dataset cell computes 2a's reference per dataset in-process.

**PR B** copies only complete cells from `output/p2f_batch1/results/` into `results/`, as in
2a and 2b, then regenerates the catalogue and coverage.

## Phase 2c point source

**Cells.** The 10 planned point-source cells (`dashboard/coverage.json`, instrument `simple`),
each with a task: 4 on `hpc_ral_cpu` and 6 on `hpc_a100`. The decisions on autolens_profiling#472:

- D1: the pin, private checkout and shared venv of 2b (`PYAUTO_PRIVATE_BASE`, default
  `/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1`; `P2C_PIN=2026.10.25.1`), with 2a's guards.
- D2: `point_source_source/source_plane jax_cpu_na_fp64@hpc_a100` runs JAX on the A100 node's
  CPU (`JAX_PLATFORMS=cpu`, `NPROC=8`) inside the A100 job. `jax_gpu_na_fp64@hpc_ral_cpu` (a GPU
  setup on `ral`, which has no GPU) got no task when the batch ran; the planner fix (#481, human
  ruling 2026-10-29 "fix the planner gap") plans a GPU setup only on a GPU host, so those two
  cells are no longer planned (the [campaign page](../../../wiki/campaigns/point_source_source/source_plane/likelihood_lifecycle_p2c.md) has the history).
- D3: no timing batch. Every evaluation is milliseconds, so one batch runs every cell under a
  fixed 1:00:00.

Each cell runs `likelihood_runtime_solved.py` (runtime) or `likelihood_breakdown.py` with
`--config-name` and `--output-path`; the scripts accept `--instrument` and always load the committed
`dataset/point_source/simple`, so `prepare_dataset` only refuses a missing dataset. Every row now
carries `device` (provenance), `resources`, `fit.png` and a `parity.json` against the eager NumPy
`AnalysisPoint(use_jax=False)` value of the same instance (default fp64 tolerance, 1e-4 x
|reference|: the eager source-plane solved lane uses a finite-difference Hessian). GPU legs set
`JAX_PLATFORMS=cuda` alone, so a node without a visible GPU fails the leg instead of writing a CPU
row.

**The runtime cell is the solved likelihood** (human ruling 2026-10-29, "Switch 2c to Solved"):
`likelihood_runtime_solved.py` times `FitPositionsImagePairAllSolved` / `FitPositionsSourceSolved`
with `PointSolved` (5 free parameters), the library default and what the breakdowns and the stale
source-plane runtime rows measure. Each submit also runs the free-centre `likelihood_runtime.py`
(`FitPositionsImagePairAll`, `FitPositionsSource`, 8 parameters) on every runtime setup into the
cell's `experiments/plain_runtime/` folder (5 legs, never current), the non-published comparison.
Batch 413192/413193 ran the submits of 2f98e01a, which had the two the other way round (plain as
the cell, solved in `experiments/solved_runtime/`); PR B moved the rows to this definition
without a rerun.

**Submits.**

| Submit | Partition | Legs | `--time` | `--mem` |
|--------|-----------|------|----------|---------|
| `batch_cpu/submit_campaign_p2c_point_source_ral_cpu_fp64` | `ral`, array `%16` | 6 (4 cells + 2 plain) | 1:00:00 | 16 GB |
| `batch_gpu/submit_campaign_p2c_point_source_a100_fp64` | `gpu`, `--gres=gpu:1`, one job | 9 in turn (6 cells + 3 plain) | 1:00:00 | 32 GB |

**Walls and memory.** The WALL-BASIS rows cite each script's process wall on the laptop
(`LAPTOP_WALLS`, the smoke runs of this branch, WSL2, `NPROC=4`), which compiled the image-plane
breakdown 3-6x slower than either RAL host in 2026-09, so they bound both. The A100 job needs the
sum of its legs, the array its slowest leg; the test checks 1:00:00 holds `WALL_HEADROOM` x that
plus 15 min. The laptop smokes peaked at about 1.1 GB resident.

**Rows** go to the gitignored `output/p2c/results/...`; logs and `walls.tsv` to
`output/p2c_logs/`. PR B copies complete cells into `results/` as in 2a and 2b, moving the six
stale rows to `history/` first.

## Phase 2e cluster (autolens_profiling#474)

Nothing here has run on RAL. The human reviews the decisions on the campaign pages
([image plane](../../../wiki/campaigns/cluster/image_plane/likelihood_lifecycle_p2e.md),
[source plane](../../../wiki/campaigns/cluster/source_plane/likelihood_lifecycle_p2e.md)) and gives
a go per batch.

**Cells.** 10 planned cluster cells in `dashboard/coverage.json`: `cluster/image_plane` (registry)
`jax_cpu_na_fp64@hpc_ral_cpu` and `jax_gpu_na_fp64@hpc_a100`; `cluster/source_plane` (registry cell
drafted in this branch, decision D2) `jax_cpu_na_fp64` on both hosts and `jax_gpu_na_fp64` on
`hpc_a100`, each with runtime and breakdown; all 10 legs run. The planner plans a GPU setup only on
a GPU host (autolens_profiling#481), so the §3.3 table's `jax_gpu_na_fp64` secondary is no longer
planned on the CPU-only `hpc_ral_cpu` (decision D4, settled).

Every leg gets `--standard-size --config-name <host>_fp64`. For the image plane that builds the
workspace cluster PointSolver the `cluster-pointsolver-speed` epic kept (100x100 @ 1.0",
precision 0.001, `magnification_threshold` 0.1, `residual_threshold` 10, all explicit; D1). The
source plane has no solver; the flag is recorded only. Runtime rows come from the shared driver
`scripts/misc/likelihood_breakdown/cluster_standard.py`, which writes the row, `parity.json` (JIT vs
eager NumPy `AnalysisPoint`, 1e-4 x |reference|) and `fit.png`. The breakdowns write their row and
`fit.png` and leave the cell's witness to the runtime.

**Submits.**

| Submit | Partition | Tasks | `--time` | `--mem` |
|--------|-----------|-------|----------|---------|
| `batch_cpu/submit_campaign_p2e_batch0_ral_cpu_fp64` | `ral`, array `%8` | 2 (image plane, batch 0) | 2:30:00 | 32 GB |
| `batch_gpu/submit_campaign_p2e_batch0_a100_fp64` | `gpu`, `--gres=gpu:1`, array `%2` | 2 (image plane, batch 0) | 2:30:00 | 32 GB |
| `batch_cpu/submit_campaign_p2e_cluster_ral_cpu_fp64` | `ral`, array `%8` | 4 | 0:30:00 | 32 GB |
| `batch_gpu/submit_campaign_p2e_cluster_a100_fp64` | `gpu`, `--gres=gpu:1`, one job | 6 in turn (2 `jax_cpu` legs on the node's CPU) | 1:00:00 | 32 GB |

**Proposed batch plan.** Batch 0 runs the four image-plane legs, which bound every cluster leg of
the same measurement and device (the source plane is the same model without the forward solve).
Then fill `BATCH0_WALLS` and `BATCH0_PEAK_RSS_KIB`, set `WALL_BASIS_STATUS = "measured"`, run
`--write` and the wall gate, and submit batch 1. Until then batch 1 refuses to run
(`P2E_ALLOW_PROVISIONAL_WALL=1` overrides it). The campaign is small, so an alternative is to skip
batch 0 and run batch 1 with the override under its conservative provisional walls; the human
chose batch 0 (D5). Done 2026-10-30 from batch 0 (RAL 413273, A100 413295; both witnesses `match`).
