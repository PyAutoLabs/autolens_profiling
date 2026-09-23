"""Harness-level candidates for the PSF convolution of the mapping-matrix cube.

Why this module exists
----------------------

Phase 1 of the ``hst-gpu-non-solver-residue`` epic (autolens_profiling#268)
traced the fused single-call production jit on the A100 (HST Delaunay N=1500,
fp64, budget 7): 31.64 ms, whose largest *computation* is the PSF convolution of
the mapping-matrix cube — 7.12 ms (22 %): a 1.99 ms scatter into the padded
``(180, 180, 1500)`` fp64 FFT frame plus 3.7 ms of ``fft`` kernels, compiled
once. Phase 3 (autolens_profiling#295) asks whether a different convolution of
that cube is cheaper **inside the same fused program**, at the campaign's 1e-9
relative pin against the unmodified library answer.

**Nothing in PyAutoArray moves.** Exactly as ``library_solver_injection`` does
for the solver, this module rebinds the library's own entry point —
``autoarray.operators.convolver.Convolver.convolved_mapping_matrix_from`` — for
the duration of a scoped context, and every other part of the likelihood stays
the library's own code on the library's own path. A candidate that wins becomes
a separate PyAutoArray prompt via /intake; ``library_change_if_wins`` states
what that change would be.

Where the patch goes, and why it takes
--------------------------------------

The inversion reaches the convolution as
``self.psf.convolved_mapping_matrix_from(mapping_matrix=..., mask=self.mask,
use_mixed_precision=..., xp=self._xp)``
(``autoarray/inversion/inversion/imaging/abstract.py:136``, and :207 / :298 for
the mapper dicts) — a method lookup on the **instance**, resolved at call time.
Rebinding the **class** attribute is therefore picked up by every Convolver,
including the one the dataset built long before the context was entered.

The Convolver is a closed-over Python constant of the traced likelihood, so the
patch acts at **trace** time: it must be installed before the jit is lowered,
and every injection context must build fresh ``jax.jit`` objects (a program
compiled under one context is cached and would be handed back unchanged).

What is delegated to the library unchanged
------------------------------------------

The candidates are JAX-path, no-blurring-matrix, no-over-sampling rewrites of
the FFT branch — the only branch the fixed-light cell's route d reaches. Every
other call is delegated to the original method, untouched, and counted under
``counts["delegated"]``:

- a NumPy ``xp`` (the eager ``FitImaging`` the cell builds, and the NumPy
  reference the unit tests compare against);
- a ``blurring_mapping_matrix`` (no inversion call site passes one today, so a
  candidate never has to reproduce the blurring-region scatter);
- ``convolve_over_sample_size > 1`` (the over-sampled helpers are a different
  program; phase 3 does not touch over-sampling);
- ``use_mixed_precision=True`` for an fp64 candidate (the fp64 levers are
  defined against the fp64 library path; the precision rows are the
  diagnostics below).

The candidates
--------------

``control``
    No patch at all — the library's FFT path as shipped. The context still
    yields a counts dict so the cell's code path is identical for every row.
``frame_pow2``
    The FFT frame rebuilt at the next power of two per axis (180 -> 256 for
    the HST cell). Bounds the frame question: cuFFT is fastest on 2^n, but
    180 = 2^2 3^2 5 is already cuFFT-friendly and the cube doubles.
``layout_src_first``
    The cube as ``(n_src, fy, fx)``: batch axis LEADING, contiguous planes, the
    transforms over ``axes=(1, 2)``, and the matching gather + transpose on the
    way out (the trailing transpose is inside the measured row).
``real_space_direct``
    The library's own ``use_fft=False`` JAX path
    (``convolved_mapping_matrix_via_real_space_from``,
    ``jax.scipy.signal.convolve(..., method="direct")``) on the same padded
    frame. A 3-D convolution with a length-1 kernel axis.
``conv_cudnn_batched``
    ``lax.conv_general_dilated`` over the cube with ``n_src`` as the BATCH axis
    (``NCHW``, one channel) — a true 2-D convolution cuDNN can batch, where the
    library's real-space path is a 3-D convolution. ``lax.conv`` is a
    cross-correlation, so the kernel is flipped on both axes and padded
    ``(k // 2, (k - 1) // 2)`` per axis to reproduce ``scipy.signal.convolve``'s
    ``mode="same"`` centring for odd and even kernels alike.
``mp_cube_c64`` (DIAGNOSTIC)
    The library FFT path with ``use_mixed_precision=True``: fp32 cube, complex64
    forward FFT, complex128 kernel multiply (what ``convolver.py:1195-1205``
    documents and the image path already ships).
``c64_full`` (DIAGNOSTIC)
    complex64 end to end, including ``state.fft_kernel_c64``; the result is cast
    back to fp64.

**Diagnostic rows may never be quoted as levers.** The campaign's constraint is
fp64 with a 1e-9 relative pin; ``fp64_exact=False`` candidates are measured for
the milliseconds they would save and the nats they cost, and nothing more.

Blurring-mask semantics (the pin is like-for-like)
---------------------------------------------------

Every candidate — and the library's FFT and real-space paths — scatters the
mapping matrix into a zero frame at the **unmasked** pixels only (no blurring
matrix is passed at any inversion call site), convolves, and gathers the result
back at the unmasked pixels. Flux blurred *out* of the mask is dropped and no
flux is blurred *in* from outside it. The frame size and padding therefore do
not change the answer, only the arithmetic; the numpy real-space path
(``convolved_mapping_matrix_via_real_space_np_from``) is the ground truth the
unit tests pin every fp64 candidate against at 1e-12.
"""

from __future__ import annotations

import contextlib
import inspect
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CANDIDATES",
    "LIBRARY_CONVOLVED_MAPPING_MATRIX_DOTTED",
    "Candidate",
    "candidate_provenance",
    "psf_convolution_injected",
]

#: The dotted name this module rebinds, recorded verbatim into the results JSON.
LIBRARY_CONVOLVED_MAPPING_MATRIX_DOTTED = (
    "autoarray.operators.convolver.Convolver.convolved_mapping_matrix_from"
)


@dataclass(frozen=True)
class Candidate:
    """One row of the phase-3 grid."""

    name: str
    label: str
    #: ``control`` | ``lever`` | ``diagnostic``.
    kind: str
    #: ``True`` when the candidate is fp64 arithmetic and must pass the 1e-9 pin.
    fp64_exact: bool
    library_change_if_wins: str


#: Insertion order is the submit array's order (task 0 = control).
CANDIDATES: dict[str, Candidate] = {
    "control": Candidate(
        name="control",
        label="library FFT path as shipped (no patch)",
        kind="control",
        fp64_exact=True,
        library_change_if_wins="none — this is the library",
    ),
    "frame_pow2": Candidate(
        name="frame_pow2",
        label="FFT frame at the next power of two per axis",
        kind="lever",
        fp64_exact=True,
        library_change_if_wins=(
            "ConvolverState.__init__: round fft_shape up to 2**ceil(log2(.)) instead of "
            "scipy.fft.next_fast_len(..., real=True) (or on GPU only)"
        ),
    ),
    "layout_src_first": Candidate(
        name="layout_src_first",
        label="cube (n_src, fy, fx), transforms over axes (1, 2)",
        kind="lever",
        fp64_exact=True,
        library_change_if_wins=(
            "Convolver.convolved_mapping_matrix_from: scatter into (n_src, fy, fx), "
            "rfft2/irfft2 over axes=(1, 2) with fft_kernel[None], gather [:, iy, ix].T"
        ),
    ),
    "real_space_direct": Candidate(
        name="real_space_direct",
        label="library real-space path (jax.scipy.signal.convolve, direct)",
        kind="lever",
        fp64_exact=True,
        library_change_if_wins=(
            "configuration only: general.psf.use_fft_default false for pixelized "
            "fits (or a per-dataset use_fft=False) — the code path already ships"
        ),
    ),
    "conv_cudnn_batched": Candidate(
        name="conv_cudnn_batched",
        label="lax.conv_general_dilated, n_src as the batch axis (2-D conv)",
        kind="lever",
        fp64_exact=True,
        library_change_if_wins=(
            "Convolver.convolved_mapping_matrix_via_real_space_from: replace the 3-D "
            "jax.scipy.signal.convolve with a batched 2-D lax.conv_general_dilated "
            "(n_src as N, flipped kernel), and route the use_fft=False path to it"
        ),
    ),
    "mp_cube_c64": Candidate(
        name="mp_cube_c64",
        label="DIAGNOSTIC fp32 cube + complex64 rfft2, complex128 kernel multiply",
        kind="diagnostic",
        fp64_exact=False,
        library_change_if_wins=(
            "none under the fp64 campaign constraint — this is "
            "Settings.use_mixed_precision=True on the mapping path, already shipped"
        ),
    ),
    "c64_full": Candidate(
        name="c64_full",
        label="DIAGNOSTIC complex64 end to end (fft_kernel_c64), cast to fp64",
        kind="diagnostic",
        fp64_exact=False,
        library_change_if_wins=(
            "none under the fp64 campaign constraint — convolver.py:1195-1205 documents "
            "why the mapping path keeps the complex128 kernel multiply"
        ),
    ),
}


def _assert_signature_covers(original, patched) -> None:
    """Fail at injection time if the library's method has grown a parameter.

    A private copy of ``library_solver_injection._assert_signature_covers``: that
    one names the solver's dotted path in its message, and a message that names
    the wrong function sends the reader to the wrong file.
    """
    _kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    library_params = {
        name for name, p in inspect.signature(original).parameters.items() if p.kind in _kinds
    }
    wrapper_params = set(inspect.signature(patched).parameters)
    missing = sorted(library_params - wrapper_params)
    if missing:
        raise TypeError(
            f"{LIBRARY_CONVOLVED_MAPPING_MATRIX_DOTTED} takes {missing}, which this harness "
            f"wrapper does not accept. The library has changed under the injection: add the "
            f"parameter to `patched` and forward it, or the injected route is a different call "
            f"from the library's own. (Do NOT change the library — this module exists so "
            f"nothing in PyAutoArray has to move.)"
        )


def _is_jax(xp) -> bool:
    return getattr(xp, "__name__", "").startswith("jax")


def _kernel_offsets(convolver) -> tuple[int, int]:
    ky, kx = convolver.kernel.shape_native
    return (ky - 1) // 2, (kx - 1) // 2


def pow2_shape(shape) -> tuple[int, ...]:
    """``2**ceil(log2(s))`` per axis."""
    return tuple(1 << int(math.ceil(math.log2(int(s)))) for s in shape)


# ---------------------------------------------------------------------------
# The candidate bodies. Each takes the Convolver, the slim (N_pix, n_src)
# mapping matrix and the mask, and returns the slim (N_pix, n_src) fp64 answer.
# ---------------------------------------------------------------------------


def _frame_pow2(convolver, mapping_matrix, mask):
    import jax.numpy as jnp

    state = convolver.state_from(mask=mask)
    shape_big = pow2_shape(state.fft_shape)
    mask_big = state.source_mask.resized_from(shape_big, pad_value=1)
    iy, ix = mask_big.slim_to_native_tuple
    n_src = mapping_matrix.shape[1]

    mm = jnp.asarray(mapping_matrix, dtype=jnp.float64)
    cube = jnp.zeros(tuple(shape_big) + (n_src,), dtype=jnp.float64).at[iy, ix].set(mm)
    # Host constant, like state.fft_kernel_mapping: the kernel's FFT at the big frame.
    kernel_fft = np.fft.rfft2(convolver.kernel.native.array, s=shape_big)[..., None]
    blurred = jnp.fft.irfft2(
        kernel_fft * jnp.fft.rfft2(cube, s=shape_big, axes=(0, 1)), s=shape_big, axes=(0, 1)
    )
    off_y, off_x = _kernel_offsets(convolver)
    blurred = jnp.roll(blurred, shift=(-off_y, -off_x), axis=(0, 1))
    return blurred[iy, ix]


def _layout_src_first(convolver, mapping_matrix, mask):
    import jax.numpy as jnp

    state = convolver.state_from(mask=mask)
    fft_shape = tuple(state.fft_shape)
    iy, ix = state.mask.slim_to_native_tuple
    n_src = mapping_matrix.shape[1]

    mm = jnp.asarray(mapping_matrix, dtype=jnp.float64)
    cube = jnp.zeros((n_src,) + fft_shape, dtype=jnp.float64).at[:, iy, ix].set(mm.T)
    kernel_fft = state.fft_kernel[None, ...]
    blurred = jnp.fft.irfft2(
        kernel_fft * jnp.fft.rfft2(cube, s=fft_shape, axes=(1, 2)), s=fft_shape, axes=(1, 2)
    )
    off_y, off_x = _kernel_offsets(convolver)
    blurred = jnp.roll(blurred, shift=(-off_y, -off_x), axis=(1, 2))
    # (n_src, N_pix) -> (N_pix, n_src). The transpose is part of the candidate.
    return blurred[:, iy, ix].T


def _real_space_direct(convolver, mapping_matrix, mask, xp):
    return convolver.convolved_mapping_matrix_via_real_space_from(
        mapping_matrix=mapping_matrix,
        mask=mask,
        jax_method="direct",
        xp=xp,
    )


def _conv_cudnn_batched(convolver, mapping_matrix, mask):
    import jax.numpy as jnp
    from jax import lax

    state = convolver.state_from(mask=mask)
    fft_shape = tuple(state.fft_shape)
    iy, ix = state.mask.slim_to_native_tuple
    n_src = mapping_matrix.shape[1]

    mm = jnp.asarray(mapping_matrix, dtype=jnp.float64)
    # NCHW: n_src images of one channel on the same padded frame the library uses.
    cube = jnp.zeros((n_src, 1) + fft_shape, dtype=jnp.float64).at[:, 0, iy, ix].set(mm.T)
    kernel = np.asarray(convolver.kernel.native.array, dtype=np.float64)
    ky, kx = kernel.shape
    # lax.conv is a cross-correlation: flip both axes to convolve. The padding
    # (k // 2, (k - 1) // 2) reproduces scipy.signal.convolve(mode="same")'s
    # centring — derived in the module docstring's sense and pinned by the tests.
    rhs = jnp.asarray(kernel[::-1, ::-1].copy())[None, None, :, :]
    blurred = lax.conv_general_dilated(
        cube,
        rhs,
        window_strides=(1, 1),
        padding=((ky // 2, (ky - 1) // 2), (kx // 2, (kx - 1) // 2)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=lax.Precision.HIGHEST,
    )
    return blurred[:, 0, iy, ix].T


def _c64_full(convolver, mapping_matrix, mask, xp):
    import jax.numpy as jnp

    state = convolver.state_from(mask=mask)
    fft_shape = tuple(state.fft_shape)
    cube = convolver.mapping_matrix_native_from(
        mapping_matrix=mapping_matrix,
        mask=state.mask,
        use_mixed_precision=True,
        xp=xp,
    )
    kernel_fft = state.fft_kernel_c64[..., None]
    blurred = jnp.fft.irfft2(
        kernel_fft * jnp.fft.rfft2(cube, s=fft_shape, axes=(0, 1)), s=fft_shape, axes=(0, 1)
    )
    off_y, off_x = _kernel_offsets(convolver)
    blurred = jnp.roll(blurred, shift=(-off_y, -off_x), axis=(0, 1))
    return blurred[state.mask.slim_to_native_tuple].astype(jnp.float64)


@contextlib.contextmanager
def psf_convolution_injected(candidate: str):
    """Rebind ``Convolver.convolved_mapping_matrix_from`` to *candidate*.

    Scoped: the original method is restored on exit, including on an exception.
    ``control`` installs no patch at all. Yields a mutable dict of call counters
    (``{"jax": n, "delegated": n}``) so the caller can assert the candidate was
    actually taken — a patch that never fires would report the library's own
    convolution under the candidate's label.
    """
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown PSF candidate {candidate!r}; expected one of {list(CANDIDATES)}")

    counts = {"jax": 0, "delegated": 0}
    if candidate == "control":
        yield counts
        return

    from autoarray.operators.convolver import Convolver

    original = Convolver.convolved_mapping_matrix_from
    fp64_exact = CANDIDATES[candidate].fp64_exact

    def patched(
        self,
        mapping_matrix,
        mask,
        blurring_mapping_matrix=None,
        blurring_mask=None,
        jax_method="direct",
        use_mixed_precision=False,
        xp=np,
    ):
        delegate = (
            not _is_jax(xp)
            or blurring_mapping_matrix is not None
            or self.convolve_over_sample_size > 1
            or (fp64_exact and use_mixed_precision)
        )
        if delegate:
            counts["delegated"] += 1
            return original(
                self,
                mapping_matrix,
                mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
                blurring_mask=blurring_mask,
                jax_method=jax_method,
                use_mixed_precision=use_mixed_precision,
                xp=xp,
            )
        counts["jax"] += 1
        if candidate == "frame_pow2":
            return _frame_pow2(self, mapping_matrix, mask)
        if candidate == "layout_src_first":
            return _layout_src_first(self, mapping_matrix, mask)
        if candidate == "real_space_direct":
            return _real_space_direct(self, mapping_matrix, mask, xp)
        if candidate == "conv_cudnn_batched":
            return _conv_cudnn_batched(self, mapping_matrix, mask)
        if candidate == "mp_cube_c64":
            return original(
                self,
                mapping_matrix,
                mask,
                jax_method=jax_method,
                use_mixed_precision=True,
                xp=xp,
            )
        if candidate == "c64_full":
            return _c64_full(self, mapping_matrix, mask, xp)
        raise AssertionError(f"candidate {candidate!r} has no body")  # pragma: no cover

    _assert_signature_covers(original, patched)
    Convolver.convolved_mapping_matrix_from = patched
    try:
        yield counts
    finally:
        Convolver.convolved_mapping_matrix_from = original


def candidate_provenance(candidate: str, convolver, mask, n_src: int | None = None) -> dict:
    """What *candidate* computes on this convolver + mask, as JSON-ready data.

    Reads the library's own ``ConvolverState`` for the shipped frame, so the
    record states the frame the control ran on and the frame the candidate used.
    """
    spec = CANDIDATES[candidate]
    state = convolver.state_from(mask=mask)
    fft_shape_shipped = [int(s) for s in state.fft_shape]
    if candidate == "frame_pow2":
        frame_used = list(pow2_shape(state.fft_shape))
    else:
        frame_used = fft_shape_shipped
    layout = {
        "control": "(fy, fx, n_src) — source axis LAST, transforms over axes (0, 1)",
        "frame_pow2": "(fy, fx, n_src) at the power-of-two frame, transforms over axes (0, 1)",
        "layout_src_first": "(n_src, fy, fx) — source axis FIRST, transforms over axes (1, 2)",
        "real_space_direct": "(fy, fx, n_src) convolved as a 3-D volume with a (ky, kx, 1) kernel",
        "conv_cudnn_batched": "(n_src, 1, fy, fx) NCHW — n_src is the conv BATCH axis",
        "mp_cube_c64": "(fy, fx, n_src) fp32 cube, complex64 rfft2, complex128 kernel multiply",
        "c64_full": "(fy, fx, n_src) fp32 cube, complex64 end to end",
    }[candidate]
    cube_dtype = "float32" if candidate in ("mp_cube_c64", "c64_full") else "float64"
    uses_fft = candidate not in ("real_space_direct", "conv_cudnn_batched")
    return {
        "name": spec.name,
        "label": spec.label,
        "kind": spec.kind,
        "fp64_exact": spec.fp64_exact,
        "library_change_if_wins": spec.library_change_if_wins,
        "patched": None if candidate == "control" else LIBRARY_CONVOLVED_MAPPING_MATRIX_DOTTED,
        "kernel_shape": [int(s) for s in convolver.kernel.shape_native],
        "source_mask_shape": [int(s) for s in state.source_mask.shape_native],
        "fft_shape_shipped": fft_shape_shipped,
        "frame_used": frame_used,
        "uses_fft": uses_fft,
        "use_fft_shipped": bool(convolver.use_fft),
        "cube_shape": (
            [int(n_src)] + frame_used
            if candidate == "layout_src_first" and n_src is not None
            else (
                [int(n_src), 1] + frame_used
                if candidate == "conv_cudnn_batched" and n_src is not None
                else (frame_used + [int(n_src)] if n_src is not None else None)
            )
        ),
        "cube_dtype": cube_dtype,
        "layout": layout,
        "blurring_semantics": (
            "unmasked pixels scattered into a zero frame, convolved, gathered back at the "
            "unmasked pixels; no blurring matrix at any inversion call site, so flux blurred "
            "out of the mask is dropped and none is blurred in — identical for every candidate"
        ),
    }
