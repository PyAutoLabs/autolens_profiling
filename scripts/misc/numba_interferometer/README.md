# numba_interferometer — the recovered numba w-tilde interferometer likelihood

A standalone, profiling-only pack that runs the numba w-tilde interferometer
likelihood PyAutoArray deleted, against **today's** PyAutoArray. Nothing here is
imported by the libraries; PyAutoArray is a read-only dependency.

Phase 1 of the `numba-interferometer-revisit` epic
(`autolens_profiling#223`). Phase 2 adds kernel variants behind the `kernel=`
switch and delivers the numba-vs-JAX verdict; phase 3 takes the preload build
itself as its own line item.

## Provenance

Everything in `inversion_interferometer_numba_util.py` is a verbatim recovery
(function bodies unchanged) from PyAutoArray commit **`0b90c401`** (2025-12-17,
*"fast chi squared implemented"*) — the last commit on `main` with the whole
numba w-tilde interferometer likelihood intact:

```bash
git -C PyAutoArray show \
  0b90c401:autoarray/inversion/inversion/interferometer/inversion_interferometer_util.py
git -C PyAutoArray show \
  0b90c401:autoarray/inversion/inversion/interferometer/w_tilde.py
git -C PyAutoArray show \
  0b90c401:autoarray/dataset/interferometer/w_tilde.py
```

The path was dismantled by PR #201 (`c9605b5b`, merged `e72defbb`, 2025-12-21)
and PR #204 (`222ee046`, merged `a655bb9e`, 2026-02-03), which replaced the numba
scatter-accumulate with the JAX/FFT `InterferometerSparseOperator` that lives at
the same path today. No PyAutoMind record documents the removal — it happened
inside two large feature PRs.

Kept: the six core kernels (lines 13-608) plus `sub_slim_indexes_for_pix_index`
(~line 1838). Dropped: the COSMA multiprocessing experiments (`parallel_*`,
`jit_loop*`, `make_2d`, the staged/chunked preload) and two pure-NumPy helpers
the pack does not use.

## The algebra is unchanged — only the application differs

With `d ∈ C^K` the visibilities, `W = diag(1/σ²)`, `F` the (NU)DFT from `N`
masked real-space pixels to `K` uv points and `M ∈ R^{N×S}` the pixelization's
**real-space** mapping matrix:

```
W~ = Re(Fᴴ W F)        the translation-invariant precision operator
d~ = Re(Fᴴ W d)        the dirty image
D  = Mᵀ d~             the data vector
F_curv = Mᵀ W~ M       the curvature matrix
χ² = sᵀ F_curv s − 2 sᵀ D + Σ d_r²/σ_r² + Σ d_i²/σ_i²
```

`W~` depends only on the pixel **offset** `(Δy, Δx)`, so it is stored as a
`(2Ny, 2Nx)` preload rather than an `N × N` matrix:

```
preload[Δy, Δx] = Σ_k σ_r[k]⁻² cos(2π [Δx·u_k + Δy·v_k])
```

This is *the same array* today's PyAutoArray calls `nufft_precision_operator`
(the recovered name was `curvature_preload`). `test_parity.py` pins that
elementwise.

What changed between `0b90c401` and `main` is only how `W~` is **applied**:

| | recovered (this pack) | `main` (`InversionInterferometerSparse`) |
|---|---|---|
| curvature assembly | numba scatter: `F[s0,s1] += preload[Δy,Δx]·w0·w1` over image-pixel pairs | FFT convolution: `Khat = fft2(preload)`, batched over source columns |
| complexity | `O(N² P²)`, `P` = mapper neighbours/pixel (3 Delaunay, 4 bilinear rectangular) | `O(B · Ny Nx log(Ny Nx))` |
| backend | numba, CPU only, no autodiff | JAX, jit/GPU/autodiff capable |
| mapper consumed as | dense `pix_indexes/sizes/weights_for_sub_slim_index`, rows on the **slim** grid | COO triplets, rows on the **extent-flat** grid, `sub_fraction` folded in |
| mixed linear objects | not supported (one mapper, no func-lists) | fully supported |

There is therefore **no missing mathematics** to re-derive.

### Caveat — `W~` has no compact support for an interferometer

For imaging, `W~` is the PSF autocorrelation, so it is zero beyond the kernel's
footprint and the equivalent imaging numba kernel only visits pixel pairs within
that footprint. For an interferometer there is no such support: every pair of
image pixels has a non-zero `W~` entry, which is exactly why the reference
kernel's double loop over image pixels is `O(N²)` with no sparsity to exploit,
and why the FFT reformulation won. Expect ~0.24 s per curvature build at `sma`
(3852 masked pixels), rising quadratically with the masked-pixel count.

### Precondition (shared with `main`)

`W~ = Re(Fᴴ W F)` is built from the **real-part** noise sigma alone, so it is
exact only when every visibility has `σ_real == σ_imag`.
`Interferometer.apply_sparse_operator` enforces that today and raises otherwise;
this pack inherits the guarantee by reading the operator's dirty image.

## What the pack contains

| File | What it is |
|---|---|
| `inversion_interferometer_numba_util.py` | The recovered numba kernels, `jit` from `autoarray.numba_util`. |
| `preload.py` | `NumbaPreload` — `curvature_preload`, `dirty_image`, `real_space_mask`, `native_index_for_slim_index`. `from_sparse_operator(dataset)` / `via_numba(dataset)`. |
| `inversion.py` | `InversionInterferometerNumba(AbstractInversionInterferometer)`. |
| `fit.py` | `numba_log_evidence_from(fit) -> (log_evidence, inversion)`. |
| `test_parity.py` | The pins, including a control that a broken kernel constant fails them. |

`InversionInterferometerNumba` **raises** (never falls back silently) on a
linear-function list, on more than one mapper, on `over_sample_size != 1`, on a
non-NumPy `xp`, and on an unknown `kernel=`. The over-sampling guard matters: the
modern sparse triplets fold `over_sampler.sub_fraction` into the mapping weights
and the recovered kernel does not, so with over-sampling the two would silently
compute different curvature matrices.

`NumbaPreload.from_sparse_operator` reads `dirty_image` straight off
`dataset.sparse_operator`, but **rebuilds** `curvature_preload` with
`nufft_precision_operator_via_np_from`, because
`InterferometerSparseOperator.from_nufft_precision_operator` keeps only
`Khat = fft2(preload)` and discards the real-space array the scatter kernel
indexes. That is a recomputation with the same function and the same arguments
`apply_sparse_operator` itself used — pinned elementwise at `rtol=1e-10`.

## How to run

From the repository root, with the worktree's `activate.sh` sourced:

```bash
# The parity pins (~10 s; auto-simulates the SMA dataset if missing).
python -m pytest scripts/misc/numba_interferometer/test_parity.py -q

# The per-step breakdowns (JSON + PNG under results/breakdown/interferometer/).
OMP_NUM_THREADS=1 python scripts/interferometer/likelihood_breakdown/delaunay_numba.py
OMP_NUM_THREADS=1 python scripts/interferometer/likelihood_breakdown/pixelization_numba.py

# The JAX/FFT comparator whose rows phase 2 ingests.
python scripts/interferometer/likelihood_breakdown/delaunay.py --instrument sma
```

Both breakdown scripts default to `--instrument sma`; `alma` works but the
reference kernel costs seconds per call there, so `N_REPEATS` drops
automatically. Each records its own `log_evidence` **and** the
`jax_reference_log_evidence` of the identical dataset and model through
`InversionInterferometerSparse`, and pins the two against each other at
`rtol=1e-6` — a divergence between them is a bug in one of the two applications
of `W~`, not in the algebra.

Using the pack directly:

```python
from numba_interferometer.fit import numba_log_evidence_from

dataset = dataset.apply_sparse_operator(use_jax=False)
fit = al.FitInterferometer(dataset=dataset, tracer=tracer, adapt_images=adapt_images, xp=np)

log_evidence, inversion = numba_log_evidence_from(fit)
```
