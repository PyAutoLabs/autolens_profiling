# Point-source shared likelihood breakdown

Date: 2026-09-20  
Issue: [autolens_profiling #291](https://github.com/PyAutoLabs/autolens_profiling/issues/291)  
Instrument commit: `3052443ca948096e9f6b2593cc352a9b0236536a`

## What landed

`scripts/point_source/likelihood_breakdown/image_plane.py` is the shared CPU/GPU
instrument for the image-plane PointSolver likelihood. It uses one production
decomposition on every JAX backend and writes the actual device, precision,
thread environment, package versions, source revisions, solver configuration,
JIT phases, step shapes and numerical controls into each result. Hardware rows
remain separate through `--config-name`; CPU timing is not treated as GPU
evidence.

The primary path is `PointSolved` with
`FitPositionsImagePairAllSolved`. A separately labelled free-centre
`PointFlux` / `FitPositionsImagePairAll` fused control prevents the two
likelihood variants from being conflated.

The solver is opened into five cumulative boundaries for each of its eight
refinement steps:

1. ray trace the current image-plane triangles;
2. find source-containing triangle indexes;
3. select the corresponding image-plane triangles;
4. construct the requested neighbourhood;
5. up-sample for the next refinement.

The source-centre solve and final magnification filter bracket those prefixes.
The final row is the difference between the last prefix and the fused production
likelihood, covering pairing chi-squared plus wrapper/fusion effects.

## CPU fp64 reference

Canonical artifact:
`results/breakdown/point_source/image_plane_local_cpu_fp64.json` and its PNG.

- Backend: JAX CPU, fp64, JAX/jaxlib 0.10.2, `NPROC=8`.
- Dataset: seeded (`noise_seed=1`) four-image `simple` positions dataset.
- Solver: 100x100 grid at 0.2 arcsec, 0.001 arcsec precision, eight
  refinements, neighbour degree 1, containing capacity 15.
- Solved fused likelihood: 62.687 ms/call; lower 11.496 s; compile 17.643 s.
- Plain fused control: 74.346 ms/call; lower 6.479 s; compile 13.291 s.
- Solved source-centre prefix: 1.209 ms/call.
- Eager/JIT likelihood: 7.743201200876817 / 7.743201200876812.
- Plain eager/JIT likelihood: 7.196577317761017 / 7.196577317761015.
- `vmap(2)` returned the solved JIT value in both lanes.
- The full model gradient was finite (`L2=1783.3021072305985`).
- The final filtered result contained four finite model positions.

The committed run took 626 s wall time because every cumulative prefix receives
its own lowering and compilation. Another workspace smoke job shared the host,
so these numbers define this row rather than a comparison with measurements from
another process or date.

## Reading the rows

The `steps` table is formed from successive differences of cumulative prefix
timings. Independent prefixes can receive different XLA fusion and scheduling,
so negative rows are retained and are not interpreted as negative work. The
absolute prefix values live in `prefix_steady_per_call_s`; the fused solved
likelihood is the authoritative end-to-end runtime. The step sum telescopes to
that fused value by construction and must not be presented as independently
additive kernel timing.

This distinction matters here: the largest observed prefix was step 6
up-sampling at 453.626 ms/call, while the fused likelihood was 62.687 ms/call.
That gap demonstrates a fusion boundary effect, not a claim that production
up-sampling alone costs 454 ms.

## GPU handoff

The A100 campaign should run this exact committed instrument with a distinct
configuration label, first on the preserved unoptimized library revisions
recorded in the CPU JSON. It must validate likelihood and finite-position parity
before using device timelines to attribute GPU kernels. Mixed precision,
batch-throughput, VRAM and kernel-trace conclusions remain GPU-task scope; this
CPU result makes none of those claims.

Example A100 invocation after updating the RAL checkout and PyAuto stack:

```bash
python scripts/point_source/likelihood_breakdown/image_plane.py \
  --config-name hpc_a100_fp64
```

The PointSolver has no `Settings` mixed-precision switch. A future
`hpc_a100_mp` row must therefore state exactly which dtype or policy changes it
applies rather than assuming `--use-mixed-precision` changes this likelihood.
