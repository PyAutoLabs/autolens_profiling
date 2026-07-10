# PreOptimizationTimes — likelihood_breakdown baseline (polish phase 4)

Status: laptop-CPU fallback tier COMPLETE (2026-07-10); A100 canonical tier
PENDING (RAL GPU nodes down; 15 dispatch-ready submits in
`hpc/batch_gpu/submit_breakdown_*`). Task: autolens_profiling#59.

## The baseline (laptop CPU fp64, policy env, quiet machine)

| Cell | Instrument | Path | Step-sum total | vs May rows | Top step |
|------|-----------|------|---------------:|------------:|----------|
| `imaging/mge` | hst | dense | 179.5 ms | 1.01× | Blurred mapping matrix (59%) |
| `imaging/pixelization` | hst | dense | 8.65 s | 1.11× | Curvature matrix F (48%) |
| `imaging/pixelization` | hst | sparse | 10.17 s | 1.14× | Curvature matrix F (46%) |
| `imaging/delaunay` | hst | dense | 10.07 s | 0.89× | Curvature matrix F (42%) |
| `imaging/delaunay` | hst | sparse | 8.81 s | 1.11× | Curvature matrix F (48%) |

Verdict vs the May (v2026.5.29.4) rows: **no library-level drift** —
0.89–1.14× is measurement scatter. The Curvature matrix (F) step is the
dominant optimization target on every mesh cell; MGE is convolution-bound.

## Cells deferred to the A100 tier (GPU-only on this laptop)

`interferometer/delaunay @ alma_high`, `datacube/delaunay @ alma_high`,
`datacube/inversion_setup_decompose @ alma_high`: the NUFFT precision-operator
construction exceeded a 2 h timeout twice under ambient load (consistent with
the phase-3 infeasibility map for alma-tier visibility counts on CPU). Per the
GPU-first platform policy these land via the A100 submits when RAL returns.

## Measurement methodology (read before comparing runs)

The canonical environment for breakdown numbers is:

- `XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"` **explicitly
  exported in the shell** — this is autoconf library policy
  (`autoconf/jax_wrapper.py` sets it at import and recommends the terminal
  export). Do not rely on the in-process set: whether XLA honours it depends
  on import-vs-backend-init order. The A100 submits export it explicitly.
- A **quiet machine**. This is not a nicety: the first pass of this campaign
  ran with ~12/15 GB in use from parallel sessions and came back **2.5–5.3×
  slow across every step uniformly** — initially misread as a library
  regression. Uniform per-step inflation (including trivially cheap steps) is
  the contention signature; targeted regressions concentrate in specific steps.
- Result JSONs record `xla_flags` / `omp_num_threads` / `cpu_count`
  (`device` dict) so env drift between runs is attributable. Note the
  recorded `xla_flags` is the env at JSON-write time — autoconf re-sets it
  in-process — so shell-level provenance is canonical.

Measured flag effect (back-to-back `likelihood_runtime/imaging/mge.py` pair,
equal load): single-JIT 0.141 s → 0.218 s (**1.54×** with the flag);
**vmap per-call insensitive** (0.110 s both ways). Constant folding only has
something to fold when inputs are concrete constants — batch tracers defeat
it, which is why the runtime package's vmap numbers are the honest
cross-config comparator regardless of the flag.

## Cross-package sanity (fusion caveat)

Step-sums vs the phase-3 runtime full-pipeline single-JIT for the same cells:
pixelization 8.65 s vs 13.7 s and delaunay 10.07 s vs 16.7 s — the runtime
numbers were taken on the 2026-07-08/09 overnight matrix and carry their own
load conditions, so treat cross-package ratios qualitatively until the A100
tier provides both packages on the same quiet hardware.
