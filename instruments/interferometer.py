"""Per-instrument interferometer dataset presets.

Single source of truth for interferometer dataset geometry + simulator
configuration. Consumed by:

- ``simulators/interferometer.py`` (re-exports ``INSTRUMENTS`` and uses every
  field to drive the simulator + the lensed-source NUFFT transformer).
- ``scripts/interferometer/likelihood_runtime/{delaunay,mge,pixelization}.py`` (read
  ``pixel_scale``, ``real_space_shape``, ``mask_radius``, ``transformer_chunk_size``).
- ``scripts/interferometer/likelihood_runtime/datacube/delaunay.py`` (same as above; per-channel).
- ``scripts/interferometer/likelihood_breakdown/*.py`` (same).
- ``vram/config.py`` (uses instrument keys to index the vmap batch_size table).

Each preset's fields:

- ``pixel_scale``             — arcsec per pixel.
- ``real_space_shape``        — (n_y, n_x) of the real-space image grid.
- ``mask_radius``             — circular mask radius in arcsec.
- ``n_visibilities``          — number of (u, v) baselines in the dataset.
- ``uv_scale``                — RNG sampling scale for (u, v) coordinates.
- ``noise_sigma``             — noise per visibility (in data units).
- ``seed``                    — RNG seed for noise + uv generation.
- ``transformer``             — ``"dft"`` or ``"nufft"`` (selects the
  transformer in both simulator and runtime).
- ``transformer_chunk_size``  — ``None`` for one-shot NUFFT, or a positive
  integer to cap the nufftax gather buffer (PyAutoArray#330). Required at
  alma_high / jvla scale.
- ``uv_wavelengths_file``     — optional, repo-relative path to a FITS file of
  real (u, v) coverage (shape ``(n_visibilities, 2)``). When set, the simulator
  uses it in place of the ``uv_scale`` Gaussian draw (``uv_scale`` is then
  ``None``); the lens + source truth and the noise model are unchanged.
"""

from __future__ import annotations

from typing import Optional

INSTRUMENTS: dict[str, dict] = {
    "sma": {
        "pixel_scale": 0.1,
        "real_space_shape": (256, 256),
        "mask_radius": 3.5,
        "n_visibilities": 190,
        "uv_scale": 3.0e5,
        "noise_sigma": 1000.0,
        "seed": 1,
        "transformer": "dft",  # 190 vis × 256² grid; DFT is cheap and exact
        "transformer_chunk_size": None,  # sma is tiny; one-shot
    },
    "alma": {
        "pixel_scale": 0.05,
        "real_space_shape": (800, 800),
        "mask_radius": 3.5,
        "n_visibilities": 1_000_000,
        "uv_scale": 2.0e6,
        "noise_sigma": 100.0,
        "seed": 1,
        "transformer": "nufft",  # 1M vis × 800² grid → DFT memory blowup; use nufftax
        "transformer_chunk_size": None,  # 1M vis × nspread²=196 ≈ 3 GB; fits A100 one-shot
    },
    "alma_high": {
        "pixel_scale": 0.025,
        "real_space_shape": (800, 800),
        "mask_radius": 3.5,
        "n_visibilities": 5_000_000,
        "uv_scale": 2.0e6,
        "noise_sigma": 100.0,
        "seed": 1,
        "transformer": "nufft",  # 5M vis × 800² grid; needs chunking via PyAutoArray#330
        "transformer_chunk_size": 1_000_000,  # caps gather buffer ~3 GB / chunk
    },
    "sdp81": {
        # Real ALMA SDP.81 uv coverage (autolens_workspace/dataset/interferometer/sdp81,
        # the public Science Verification continuum export, 108,384 vis) on the alma
        # grid, with the standard simulated lens + source: N_vis is the only thing
        # that changes between the sdp81 and alma rows (autolens_profiling#356).
        "pixel_scale": 0.05,
        "real_space_shape": (800, 800),
        "mask_radius": 3.5,
        "n_visibilities": 108_384,
        "uv_scale": None,  # real coverage, see uv_wavelengths_file
        "uv_wavelengths_file": "instruments/uv_coverage/sdp81_uv_wavelengths.fits",
        "noise_sigma": 100.0,
        "seed": 1,
        "transformer": "nufft",  # 1e5 vis × 800² grid; nufftax as at alma
        "transformer_chunk_size": None,  # one-shot
    },
    "jvla": {
        "pixel_scale": 0.01,
        "real_space_shape": (800, 800),
        "mask_radius": 3.5,
        "n_visibilities": 25_000_000,
        "uv_scale": 2.0e6,
        "noise_sigma": 100.0,
        "seed": 1,
        "transformer": "nufft",  # 25M vis stretch test; mask_radius=3.5/0.01 = 350-px radius (700-px mask diameter)
        "transformer_chunk_size": 1_000_000,  # 25 chunks × ~3 GB gather buffer each
    },
}


TRANSFORMER_CLASS_NAME: dict[str, str] = {
    "dft": "TransformerDFT",
    "nufft": "TransformerNUFFT",
}


def mask_radius_pixels(instrument: str) -> int:
    """Mask radius in pixels = ``mask_radius_arcsec / pixel_scale``."""
    cfg = INSTRUMENTS[instrument]
    return int(round(cfg["mask_radius"] / cfg["pixel_scale"]))


def transformer_chunk_size_for(instrument: str) -> int | None:
    """Per-instrument NUFFT chunk_size (None for one-shot)."""
    return INSTRUMENTS[instrument].get("transformer_chunk_size")
