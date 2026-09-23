# HST GPU residue phase 3 — the PSF convolution of the mapping-matrix cube

## Verdict

**No fp64 lever.** On the A100, the library's own FFT convolution of the padded
`(180, 180, 1500)` fp64 mapping-matrix cube is the fastest fp64 implementation measured. Five fp64
programs were run inside the same fused whole-call jit. All five pass the pre-registered `1e-9`
gate, and none beats the shipped path:

- **`layout_src_first` is a tie.** It measures 31.79 ms against control's 31.66 ms (+0.13 ms,
  +0.4 %). Its traced convolution row matches control's (7.111 vs 7.114 ms). The difference sits
  inside the run-to-run spread. On the command-buffers-on basis the sign even flips: 31.62 ms
  against control's 32.46 ms. Control's own two whole-call readings in this job differ by 0.79 ms,
  and route-d controls across phases 1-3 span 31.03-31.66 ms.
- **`frame_pow2` is slower and doubles memory.** Padding the frame to 256×256 costs +4.52 ms on
  the call (+4.02 ms in the convolution row), and the process high-water mark rises from 1.18 to
  2.38 GB. Compile time does *not* lengthen on the A100 (route d 4.47 s against 4.35 s). The
  tripled compile seen in RTX screening does not carry over.
- **Real-space convolution is 1.5× and 3.8× slower.** Both real-space rows lower to a cuDNN
  convolution. The batched 2-D form (`conv_cudnn_batched`) costs 45.99 ms (1.45×). The library's
  own `use_fft=False` path (`real_space_direct`, a 3-D conv with a length-1 axis) costs 120.97 ms
  (3.82×), with 96.2 ms spent in one `cudnn-conv` kernel.

The issue's lever threshold was ≥ 1.5 ms faster at the pin on every fp64 leg. No fp64 candidate
comes close, so no PyAutoArray prompt is filed.

**Only the two DIAGNOSTIC precision rows are faster.** `mp_cube_c64` (fp32 cube, complex64
forward FFT, complex128 kernel multiply) saves 2.14 ms (6.7 % of the call). `c64_full` (complex64
end to end) saves 3.98 ms (12.6 %). Both fail the campaign's `1e-9` pin by two to three orders of
magnitude:

| row | fiducial vs route b | max over 8 draws vs route b |
|---|---|---|
| `mp_cube_c64` | 4.51e-9 (1.3e-4 nats) | 1.39e-7 (1.3e-3 nats) |
| `c64_full` | 2.34e-8 (6.8e-4 nats) | 4.41e-7 (3.8e-3 nats) |

Under this campaign's fp64 constraint they are **not levers and may not be quoted as such.**

They remain a **human decision point**, not a profiling question: would we accept ~1e-3 nats per
evaluation for 7-13 % of the call? If that is ever taken up as a separate prompt, it must
re-measure on a full inference, not on this fixed-light cell. The convolver's own comment
(PyAutoArray `22e6d608`, `autoarray/operators/convolver.py:1198-1210`) records that fp32 end to
end drifted the figure of merit by O(1) on the `delaunay_mge` regression. That drift is why the
shipped mixed-precision path keeps the kernel multiply in complex128.

## Experiment

- Job: RAL SLURM array `350573`, tasks 0-6, one A100 80GB PCIe each. Submitted 2026-09-23
  09:50:53 UTC. Tasks 0-3 ran on `euclid-ral-gpu-1` and tasks 4-6 on `euclid-ral-gpu-2`. Every task
  was `COMPLETED 0:0` in 59-69 s, and every footer shows `AUTOTUNE_ENTRIES count=0` and
  `CELL_EXIT=0`. Each `.err` holds only the benign `mask_2d_util.py:564` padding warning.
- Profiling source: `feature/hst-gpu-residue-p3` @ `e2b46f3`. Every JSON's `cell_source_sha256`
  (`acc9ccdd…`) equals the sha256 of `fixed_light_trace.py` at that commit.
- Library revisions: PyAutoNerves `1fa613aa`, PyAutoFit `a7368401`, PyAutoArray `22e6d608`,
  PyAutoGalaxy `70a61e26`, PyAutoLens `2aaa1c1a`. These are the phase-2 revisions. The dataset
  sha256s are also identical to phase 2's.
- Likelihood: HST imaging, Delaunay N=1500, fp64, fixed lens light, dense inversion, border
  relocator on, certified active-set harness injection at the production budget 7. The certified
  solver is not shipped production code.
- Mode: `--psf-candidate <name>`. The candidate convolution is injected into route d inside the
  fused `jax.jit`, via a class-attribute rebind of `Convolver.convolved_mapping_matrix_from`
  (`scripts/misc/likelihood_breakdown/psf_cube_injection.py`). Route b is the unmodified library
  (library PDIP plus library convolution) and is the pin reference. Injection counts confirm that
  every non-control row ran the candidate: `route_d_jax = 2`, `traced_program_jax = 2`, zero
  delegated.
- Whole-call ms: route d `timing.jit_profile`, 10 steady calls, the same basis as the phase-2
  scalar controls. The traced program (command buffers on and off, one lowering) is recorded
  separately.
- Fresh JAX compilation cache per task, with zero autotune entries at start.

## Matched table

A100, fp64, HST Delaunay N=1500, budget 7. The whole-call ms is route d with the candidate
injected. The convolution column is the traced `psf_convolution_mapping_matrix` stage row
(median of 10 calls, command-buffers-off program). The scatter column is the `input_scatter_fusion`
instruction inside that row. Compile time is route d. Peak is the process high-water mark after
all routes.

| candidate | kind | whole-call ms | Δ vs control | convolution ms | of which scatter ms | cmd-buf on ms | compile s | peak GB | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `control` | control | **31.66** | — | 7.11 | 1.99 | 32.46 | 4.35 | 1.18 | PASS (1/1) |
| `layout_src_first` | lever | 31.79 | +0.13 (+0.4 %) | 7.11 | 1.99 | 31.62 | 4.60 | 1.18 | PASS (9/9) |
| `frame_pow2` | lever | 36.18 | +4.52 (+14.3 %) | 11.14 | 2.33 | 35.98 | 4.47 | **2.38** | PASS (9/9) |
| `conv_cudnn_batched` | lever | 45.99 | +14.33 (1.45×) | 20.73 | 1.99 | 45.88 | 4.35 | 1.18 | PASS (9/9) |
| `real_space_direct` | lever | 120.97 | +89.31 (3.82×) | 96.24 | — | 120.71 | 4.28 | 1.18 | PASS (9/9) |
| `mp_cube_c64` | DIAGNOSTIC | 29.53 | −2.14 (−6.7 %) | 5.27 | 0.82 | 29.48 | 4.30 | 1.18 | DIAGNOSTIC |
| `c64_full` | DIAGNOSTIC | 27.68 | −3.98 (−12.6 %) | 3.70 | 0.83 | 27.72 | 4.27 | 1.18 | DIAGNOSTIC |

The lever term moves as expected. In every row the whole-call change follows the convolution
row's change to within ~0.5 ms: frame_pow2 +4.02 ms convolution against +4.52 ms call;
cuDNN batched +13.61 against +14.33; real space +89.12 against +89.31; the diagnostics −1.84
against −2.14 and −3.42 against −3.98.

Notes on individual rows:

- `real_space_direct` has no scatter instruction in its row: the whole row is one `cudnn-conv`
  over `f64[1,1,180,180,1500]`.
- `layout_src_first` gets the same three FFTs over a `(1500, 180, 180)` cube, with kernel times
  identical to control's. On the A100, XLA already lays out the shipped `(180, 180, 1500)`
  transform efficiently.
- Mixed fusions that span the PSF stage and another stage are 0.95-2.06 ms per row. They are
  listed per candidate in the JSONs and are not summed into the convolution column. The largest is
  `curvature_matrix_F + psf_convolution_mapping_matrix` (0.80-1.22 ms) on every FFT and cuDNN-batched
  row. On `real_space_direct` the largest is `mapping_matrix + psf_convolution_mapping_matrix`
  (0.99 ms).

Route b (the unmodified library, identical in every task) measured 50.69-50.97 ms across the seven
tasks. That 0.28 ms band is the within-job spread of an unchanged program. The layout tie also
sits inside it.

**Control reproducibility.** Compare like with like. Phase 1's quoted **31.64 ms** is the
*command-buffers-on traced executable*; this job's matching number is 32.46 ms (+2.6 %). Phase 1's
route-d `jit_profile` was 31.49 ms, against this job's 31.66 ms (+0.5 %), and its route b was
50.56 ms against 50.84 ms. Phase 2's scalar-jit controls were 31.03-31.31 ms. The production
control reproduces within 2.6 % on either basis, inside the plan's ~3 % expectation.

## Numerical gate

The gate was **pre-registered** at commit `e2b46f3`, 2026-09-23 09:25:08 UTC. That is 26 minutes
before the array was submitted, and the gate was restated on the issue before the first task
finished. The rules, all at `EQUIVALENCE_RTOL = 1e-9` relative:

1. The fiducial pin (route d with the candidate vs route b, the unmodified library) is **gated**
   for every fp64 row, control included.
2. A lever's eight seeded draws (seed 0) are **gated on `rel_diff_vs_d_control`**: route d
   against route d compiled *without* the candidate. Both use the same certified solver, so this
   pin isolates the convolution. The draw's difference vs route b is recorded, not gated.
3. Control's draws are recorded vs route b and not gated (there is no d-control).
4. Diagnostics are DIAGNOSTIC: never gated, never PASS.

The gate was set this way because phase 2 recorded a 2.0-2.6e-9 residual on seed-0 draw 7. Gating
every draw vs route b could have failed every task, control included, for a reason unrelated to
the convolution. Nothing was relaxed after seeing data.

| candidate | fiducial rel vs route b | max draw rel vs d-control (gated for levers) | max draw rel vs route b (recorded) | status |
|---|---:|---:|---:|---|
| `control` | 7.01e-11 | — | 2.48e-10 | PASS |
| `layout_src_first` | 5.66e-12 | 1.94e-10 | 2.80e-10 | PASS |
| `frame_pow2` | 7.20e-11 | 2.52e-10 | 1.47e-10 | PASS |
| `conv_cudnn_batched` | 7.40e-11 | 4.72e-10 | 1.85e-10 | PASS |
| `real_space_direct` | 9.75e-11 | 3.29e-10 | 1.27e-10 | PASS |
| `mp_cube_c64` | 4.51e-9 | 1.38e-7 | 1.39e-7 | DIAGNOSTIC |
| `c64_full` | 2.34e-8 | 4.41e-7 | 4.41e-7 | DIAGNOSTIC |

**The phase-2 draw-7 residual did not reappear.** Every fp64 draw in this job is within
`2.80e-10` of scalar route b, so the stricter every-draw-vs-b rule would have given the same
verdict. The pre-registration was conservative, not outcome-changing.

The draws are the same vectors phase 2 used (identical seed-0 offsets). On draw 7, control route
d reads −6381.7496164 and scalar route b reads −6381.7496170, a relative difference of 1.05e-10.
Phase 2's library-PDIP reference was the *vmapped* program, −6381.7496328. That value differs
from this job's scalar route b by 2.47e-9.

The residual therefore sits between compositions (`jit(vmap)` against scalar `jit`), not in the
scalar certified solver. This matches phase 2's own conclusion that the difference is a
composition-dependent fp64 reduction difference, and it is evidence for the parked
batching-reproducibility study. It is recorded here, not acted on.

## Trace diagnostics

- **Reconciliation:** within −0.11 % to −0.50 % on all seven tasks (gate ±5 %).
- **Unjoined kernel time:** 0.000 ms everywhere.
- **Profiler overhead:** 0.98-4.37 % of the untraced wall on six tasks and 5.8 % on `c64_full`.
  Overhead is not gated. It belongs in neither the stages nor the idle row, and the reconciliation
  above is against the traced wall.
- **HLO census:** status `ok` on every task. The PSF anchors (`convolved_mapping_matrix_from`,
  `_convolved_mapping_matrix_over_sampled_jax_from`, `operated_mapping_matrix_list`) were verified
  against the installed PyAutoArray `22e6d608`. The census shows that the convolution is compiled
  once in every row:
  - `fft_mapping_matrix = 3` on the five FFT rows, with `conv_mapping_matrix = 0`;
  - `conv_mapping_matrix = 1` on the two real-space rows, with zero FFTs.
- **Census rows excluded:** `rows_excluded` lists `curvature_reg_add_nn`,
  `curvature_reg_reduced_gathers_397` and `edge_subset_gathers_613`. Their `abstract.py` anchors
  have moved in `22e6d608` (for example, `curvature_reg_matrix` now spans 368-388, not line 371).
  They do not touch the PSF rows. Re-anchoring them is a follow-up and is not done here.
- **Device idle** is 7.0-7.4 ms on the candidate rows and 8.15 ms on control. The host qhull span
  is 4.61-4.96 ms on every task, so no candidate changes the callback round-trip.

## Errata carried

- **Cube frame.** The mapping-matrix cube is `(180, 180, 1500)` fp64 for the masked 141×141
  dataset (fft_shape 180, source axis last), about 389 MB. That is consistent with phase 1's 389 MB
  but not with the `(200, 200, 1500)` frame this phase's plan and issue assumed. The phase-1 note
  carries a dated erratum.
- **Stale convolver docstring.** The note at `convolver.py:65-66` ("even FFT sizes are currently
  incremented to odd sizes") no longer describes the shipped behaviour: the measured frame is the
  even 180. This is a library documentation follow-up and is not edited here.
- **JAX platform.** `JAX_PLATFORMS=cuda,cpu` alone silently runs on CPU under the workspace shell
  (jax 0.10.2). The first RTX control ran on CPU and gave an empty trace (reconciliation −100 %).
  Forcing CUDA needs `JAX_PLATFORM_NAME=cuda`, which the A100 submit sets.

## What this means for the campaign

This paragraph records where the campaign stands; it does not take a decision. Of the non-solver
terms phase 1 measured, four are now closed or parked:

- the qhull callback idle, parked behind the batching reproducibility study that phase 2 and this
  job's draw-7 comparison both point at;
- the `A.T A` F GEMM, closed as the fp64 dense floor;
- the mixed fusions, closed as the attribution's resolution limit;
- the PSF convolution of the mapping-matrix cube, closed with no fp64 lever.

The remaining listed lever is the second Cholesky (0.89 ms, phase 4). The precision row is a
human policy decision, not a profiling question.

## Provenance and historical separation

The seven A100 JSON/PNG pairs are
`results/breakdown/imaging/fixed_light_trace_delaunay_psf_<candidate>_hpc_a100_fp64_fixed_light_trace.{json,png}`.
Their sha256s, SLURM per-task state, footers and the gate decision are in
[`hst_gpu_residue_phase3_job350573.json`](hst_gpu_residue_phase3_job350573.json).

RTX 2060 rows were screening only: they checked the harness's pins, reconciliation, census and
exit codes, and they are never ranked because of the GeForce fp64 rate. The artifacts are
[`fixed_light_trace_delaunay_psf_<candidate>_local_rtx2060_fp64_fixed_light_trace.json`](../breakdown/imaging/fixed_light_trace_delaunay_psf_control_local_rtx2060_fp64_fixed_light_trace.json)
(control linked; the six siblings share the pattern). Their `cell_source_sha256` predates the
docstring correction (`cbb2078`) and the gate commit (`e2b46f3`).

Phase 1 ([`hst_gpu_residue_phase1_2026_09.md`](hst_gpu_residue_phase1_2026_09.md)) and phase 2
([`hst_gpu_residue_phase2_vmap_2026_09.md`](hst_gpu_residue_phase2_vmap_2026_09.md)) are
unchanged apart from phase 1's erratum block. No PyAutoArray, PyAutoLens or PyAutoFit source was
changed.
