# Fixed-light profiling: current findings and experiment contracts

Status as recorded on **2026-09-18**; the GPU section adds the phase-3 verdict on
**2026-09-23**. This page supersedes obsolete Next sections,
not the historical measurements themselves. No new timings were collected for it.

## CPU: complete

The fixed-lens-light-numba-cpu epic is closed after phases 1–6, including 4b/5b.
**Phase 7 is explicitly shelved.** Assume zero dependable memo benefit under
changing sampler order for current planning. Production memo defaults were not
changed; further memo-policy/order research is outside the agreed scope.

| Finding | Scope and evidence |
|---|---|
| 932.4 → 459.2 ms (2.03x) | HST Delaunay N1500, fp64, single-thread numba sparse imaging assembly, memo OFF in both arms. Fixing lens light changes what is held fixed; preparation is separate. [Measurement](fixed_lens_light_numba_2026_09.md). |
| 413.3 → 230.0 ms (1.80x) | Same source-only family, memo ON, three CPU library optimizations. Do not multiply this by 2.03x. A100 controls unchanged. [Levers](fixed_lens_light_levers_2026_09.md). |
| No further curvature/permutation lever | Whole-call tests rejected the candidates. Keep the current kernels and positive-only solve. [Phase 4](fixed_lens_light_s4_2026_09.md). |
| Memo precheck: NO_LEVER | Numerical checks passed, independent performance targets failed. No policy promotion. [Phase 5b](fixed_lens_light_numba_memo_policy_2026_09.md). |
| NNLS dominates large systems | N4000 cold: 2.138 s nearby / 4.312 s broad; about 81% / 88% NNLS. Conditional source-only synthetic sequences, not sampler throughput. [Scaling](fixed_lens_light_numba_scaling_2026_09.md). |

PyAutoArray [#553](https://github.com/PyAutoLabs/PyAutoArray/pull/553),
[#554](https://github.com/PyAutoLabs/PyAutoArray/pull/554) and
[#555](https://github.com/PyAutoLabs/PyAutoArray/pull/555) are recorded as merged
pending release. Research closure does not clear their release obligations or
establish release readiness. The standalone operated-mapping-matrix cache issue
and the ongoing GPU work are separate.

### Cumulative figure: provenance limit

The per-lever A/B artifacts are committed. The lever-3 note additionally uses
job **343355**'s 16-repeat control (**268.681 ms**, versus lever 2's **267.448 ms**)
as a bridge between runs. That control's JSON/PNG were deliberately left in
untracked `output/ral_job343355_16repeats/` after the feature arm failed the
overhead gate. They were not found in the local output tree during this review;
no cluster recovery was attempted and no replacement artifact was fabricated.

Thus the exact historical bridge is not reproducible from the committed tree
alone. The 1.80x remains a reported campaign comparison, with that limitation;
the committed individual A/B gains do not depend on presenting the failed job
as a complete pair. If the original is recovered, retain its source/job evidence
and label it **control-only, incomplete A/B job**. Do not rerun it merely to fill
the archive. For future headline support, version the supporting control along
with the existing source/job sidecar, including incomplete-job status.

## GPU: corrected baseline, batching experiment inconclusive, PSF convolution has no fp64 lever

The [budget-7 single-call trace](hst_gpu_residue_phase1_2026_09.md) supersedes two
claims inherited from the completed JAX fixed-light campaign:

- **25.39 ms was budget 2**; the measured budget-7 A100 Delaunay call was
  **31.64 ms** with command buffers on. The certified solver is still a harness
  injection, not a shipped library implementation.
- The attributed **~13.9 ms mesh/mapper/weights** bucket measured **0.37 ms**.
  Host callback round trips and the mapping-cube PSF convolution are material
  targets. Trace overhead and command-buffer differences remain qualified in
  the note; scaled stage times assume uniform overhead.

The older vmap scaling curve measured the solver only. Single-call timings do
not determine production whole-likelihood batching policy. Issue
[#273](https://github.com/PyAutoLabs/autolens_profiling/issues/273) is the matched
vmap-versus-scalar-jit experiment. Array **343376** measured the retired
`jax.vmap(jax.jit(fn))` composition and is retained as historical evidence only.
The replacement array **344635** measured current production
`jax.jit(jax.vmap(fn))`: fallback-on batching was slower per lane at B=4, 8 and
16, while the faster fallback-off B16 row was diagnostic only. The required
three-way `1e-9` numerical gate failed on one or more distinct lanes at B=8 and
B=16, so the [phase-2 note](hst_gpu_residue_phase2_vmap_2026_09.md) records an
explicit inconclusive policy verdict. No batching-policy or batch-aware callback
change follows from this fixed-N grid.

Phase 3 (issue [#295](https://github.com/PyAutoLabs/autolens_profiling/issues/295),
A100 array **350573**) settles the PSF convolution of the mapping-matrix cube: **no
fp64 lever**. The library's FFT convolution of the padded `(180, 180, 1500)` cube
is the fastest fp64 implementation measured inside the fused whole-call jit. Every
fp64 candidate passed the pre-registered `1e-9` gate. `layout_src_first` ties
(+0.13 ms, inside the run-to-run spread). `frame_pow2` is +4.52 ms and doubles peak
memory. The two real-space rows are 1.45× and 3.82× slower. The two faster rows,
fp32-cube and full complex64 (−2.14 / −3.98 ms), are **DIAGNOSTIC**: they miss the pin
by ~1e-3 nats on the draws. They may not be quoted as levers, and accepting them
is a human policy decision, not an open profiling question. The
[phase-3 note](hst_gpu_residue_phase3_psf_2026_09.md) has the matched table. The
remaining listed GPU lever is the second Cholesky (0.89 ms, phase 4).

The [matrix-free verdict](matrix_free_pixelized_2026_09.md) remains a no-go on its
measured grid; this summary does not reopen it or infer an unmeasured crossover.

## Contracts for the next measurement

- Fixed-light cells finish staged shared/local CLI parsing with
  `ProfileCLI.parse_cell_args`: unknown flags and abbreviated options fail.
  An older trace cell now rejects unsupported batching flags instead of silently
  producing a single-call result. Unrelated legacy cells retain their parser.
- Every row in the numba whole-call cell resets after warm-up, primes instance 0
  untimed and starts the timed stream at instance 1. Stream length and metadata
  must match before rows can be published together. This extends phase 4b's
  alignment; it does not rewrite old timing artifacts or change library defaults.
- Preserve clean/diagnostic separation, numerical equivalence, explicit thread
  settings, source/dataset hashes and missing-cell status. Use the later phase-5/6
  source/job sidecars; no new general result-schema framework is required.
- Compare within the declared device/backend, precision, solver/fallback budget,
  mesh and actual source size, regularization, light-preparation convention and
  draw sequence. Keep preparation and instrumentation costs explicit. Neither
  likelihood speedups nor harness RSS establish end-to-end sampler performance
  or production memory per worker.

For phase-2 integration, retain that branch's local batching options and replace
only its final `_cell_parser.parse_known_args()` with
`_cli.parse_cell_args(_cell_parser)` alongside the shared helper. This task does
not import or modify the unmerged batching implementation.
