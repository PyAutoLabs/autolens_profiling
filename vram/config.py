"""Per-(dataset, model, instrument) vmap batch_size table for A100 80 GB.

Populated empirically from ``vram.probe`` runs on the RAL A100 cluster
(2026-05-24). Each value is the recommended batch_size for
``jax.jit(jax.vmap(likelihood))`` on an NVIDIA A100 80 GB PCIe, with:

- 65 GB effective VRAM budget (15 GB headroom for JAX runtime, driver
  allocations, and fragmentation).
- 1.15× safety factor on the per-replica static memory estimate.
- Cap at 64 for XLA compile-time tractability.

A value of ``None`` means vmap is **intentionally skipped or blocked**
for that cell:

- Datacube cells: natural batching axis is "channels", not "parameters".
- Interferometer mge at ALMA+ scale: inherently blocked. MGE's mapping
  matrix is fully dense (every Gaussian maps to every pixel), so the
  per-call NUFFT cost is O(N_vis × N_src) and can't use the
  sparse-operator shortcut. ``transform_mapping_matrix`` chunking would
  cap per-chunk memory but per-call time remains prohibitive at 1M+ vis.
- Interferometer pixelization at ALMA+ scale: blocked on
  ``transform_mapping_matrix`` not being chunked. Pixelization's mapping
  IS sparse (localized mesh) and could eventually use the
  sparse-operator path (like delaunay); just not implemented yet.

Keys are ``(dataset, model, instrument)`` tuples — flat for easy lookup.

To add a new instrument:
  1. Run the probe SLURM script for it (see ``vram/README.md``).
  2. Read the ``recommended_batch_size`` from the probe JSON.
  3. Add the row below.
  4. Re-run the regular profile to confirm the chosen batch holds at
     steady state.
"""

from __future__ import annotations

from typing import Optional

VMAP_BATCH: dict[tuple[str, str, str], int | None] = {
    # =========================================================================
    # Imaging cells — 4 instruments, 3 cells.
    # Per-replica cost dominated by mapping_matrix (n_mask × n_source × 8 bytes).
    # AO (700-px mask) is the most constrained; euclid (70-px) the cheapest.
    # =========================================================================
    #
    # delaunay (1500-node Hilbert mesh)
    # NOTE: probe-recommended sizes halved for hst/jwst/ao after cuFFT
    # scratch-allocator failures at the probe-predicted batch. The static
    # memory_analysis() doesn't account for cuFFT batched-plan scratch.
    ("imaging", "delaunay", "euclid"): 64,  # 270 MB / replica — probe OK
    ("imaging", "delaunay", "hst"): 16,  # 922 MB / replica — probe said 62, cuFFT failed
    ("imaging", "delaunay", "jwst"): 8,  # 2,415 MB / replica — probe said 23, cuFFT failed
    ("imaging", "delaunay", "ao"): 1,  # 17,485 MB / replica — probe said 3, OOM at 3
    #
    # pixelization (35×35 = 1225-node rectangular mesh)
    ("imaging", "pixelization", "euclid"): 64,  # 273 MB / replica — probe OK
    ("imaging", "pixelization", "hst"): 16,  # 931 MB / replica — probe said 62, cuFFT failed
    ("imaging", "pixelization", "jwst"): 8,  # 2,428 MB / replica — probe said 23, cuFFT failed
    ("imaging", "pixelization", "ao"): 1,  # 17,537 MB / replica — probe said 3
    #
    # mge (~25 analytical Gaussians — small, constant per-replica cost)
    ("imaging", "mge", "euclid"): 64,  #   6 MB / replica
    ("imaging", "mge", "hst"): 64,  #  16 MB / replica
    ("imaging", "mge", "jwst"): 64,  #  42 MB / replica
    (
        "imaging",
        "mge",
        "ao",
    ): None,  # 296 MB / replica — vmap CORRECTNESS bug: 3 distinct log_ev in batch=64. Separate investigation.
    #
    # =========================================================================
    # Interferometer cells — 4 instruments, 3 cells (mge/pix blocked at ALMA+).
    # Delaunay uses the W-Tilde sparse-operator path (per-call cost is
    # mask-FFT-dominated, NOT visibility-count-dominated). mge/pixelization
    # use the full NUFFT mapping matrix (blocked at 1M+ vis — see note above).
    # =========================================================================
    #
    # delaunay (1000-node Hilbert mesh, sparse-operator path)
    ("interferometer", "delaunay", "sma"): 64,  #   92 MB / replica — probe OK
    ("interferometer", "delaunay", "alma"): 64,  #  322 MB / replica — probe OK
    (
        "interferometer",
        "delaunay",
        "alma_high",
    ): 16,  # 1,243 MB / replica — probe said 46, OOM at runtime
    ("interferometer", "delaunay", "jvla"): 3,  # 7,689 MB / replica — probe said 7, OOM at runtime
    #
    # mge — sma only; alma+ INHERENTLY blocked (dense model, O(N_vis × N_src)).
    ("interferometer", "mge", "sma"): 64,  # 160 MB / replica
    (
        "interferometer",
        "mge",
        "alma",
    ): None,  # blocked: dense mapping → 62 GB gather buffer at 1M vis
    ("interferometer", "mge", "alma_high"): None,
    ("interferometer", "mge", "jvla"): None,
    #
    # pixelization — sma only; alma+ blocked on unchunked transform_mapping_matrix.
    # Pixelization mapping IS sparse — could eventually use sparse-operator path.
    ("interferometer", "pixelization", "sma"): 64,  # 93 MB / replica
    (
        "interferometer",
        "pixelization",
        "alma",
    ): None,  # blocked: needs sparse-operator or chunked NUFFT mapping
    ("interferometer", "pixelization", "alma_high"): None,
    ("interferometer", "pixelization", "jvla"): None,
    #
    # =========================================================================
    # Datacube — intentionally skipped (parameter-axis vmap not meaningful;
    # cube batching is over channels, handled by the per-channel loop).
    # =========================================================================
    ("datacube", "delaunay", "sma"): None,
    ("datacube", "delaunay", "alma"): None,
    ("datacube", "delaunay", "alma_high"): None,
    #
    # =========================================================================
    # Point source — tiny per-replica cost, cap at 64.
    # =========================================================================
    ("point_source", "image_plane", "simple"): 64,  # 3 MB / replica
    ("point_source", "source_plane", "simple"): 64,  # <1 MB / replica
}


# Sparse-operator (w-tilde) rows. The sparse inversion path has a different
# per-replica memory profile than dense mapping, so it gets its own table
# rather than a fourth key element (keeps VMAP_BATCH's shape stable for the
# many existing readers). Unprobed sparse cells fall back to the dense row —
# conservative, since the sparse path's per-replica footprint is smaller for
# every cell measured so far. Populated by the probe-only SLURM submits
# (``hpc/batch_gpu/submit_probe_*``) as campaign probes come in.
VMAP_BATCH_SPARSE: dict[tuple[str, str, str], int | None] = {}


# Where each table's numbers came from — checked before a campaign trusts
# them. Update whenever probe results are ingested.
PROVENANCE: dict[str, str] = {
    "VMAP_BATCH": (
        "probed 2026-05-24 on RAL A100 80GB PCIe (PyAutoLens ~2026.5.x); "
        "manual halvings for cuFFT scratch failures noted per row"
    ),
    "VMAP_BATCH_SPARSE": "unprobed — falls back to dense rows",
}


def vmap_batch_for(dataset: str, model: str, instrument: str, path: str = "dense") -> int | None:
    """Return the per-(dataset, model, instrument) vmap batch_size for A100.

    ``path`` selects the inversion path: ``"dense"`` (mapping-matrix, the
    default) or ``"sparse"`` (w-tilde sparse operator). Sparse cells without
    a probed row fall back to the dense value.

    Returns ``None`` when vmap is intentionally skipped (cube cells),
    blocked (interferometer mge/pix at ALMA+), or when the cell hasn't
    been probed yet. Callers should default to a small fallback (typically
    3) when ``None`` is returned for an un-probed cell, and skip vmap
    entirely when ``None`` is returned for a known-blocked or known-skipped
    cell.
    """
    key = (dataset, model, instrument)
    if path == "sparse" and key in VMAP_BATCH_SPARSE:
        return VMAP_BATCH_SPARSE[key]
    return VMAP_BATCH.get(key)


# On non-GPU backends the vmap row is a correctness/timing sample, not a
# production configuration — the table and the probe budget are both
# A100-oriented, and batch 64 on a laptop CPU exhausts host RAM (found by
# the phase-2 local validation sweep, autolens_profiling#54).
_NON_GPU_BATCH_CAP = 3


def resolve_vmap_batch(
    dataset: str,
    model: str,
    instrument: str,
    output_dir=None,
    path: str = "dense",
    backend: str | None = None,
) -> tuple[int | None, str]:
    """Resolve the vmap batch_size, preferring a fresh probe over the table.

    Resolution order (returns ``(batch_size, source)`` where ``source``
    says which step won, for the run log):

    1. ``<output_dir>/vmap_probe_<model>[_sparse].json`` — written by a
       ``--vmap-probe`` run (Phase A of the A100 submit scripts). Used only
       if its ``dataset``/``model``/``instrument`` fields match this cell,
       so a stale table can never OOM a job whose submit re-probed.
    2. The curated table (``vmap_batch_for``, including the sparse
       fallback-to-dense rule).

    ``backend`` is the running JAX backend (``jax.default_backend()``).
    The table and the probe budget are A100-oriented, so on any non-``gpu``
    backend the resolved batch is clamped to ``_NON_GPU_BATCH_CAP`` (an
    intentional ``None`` stays ``None``).

    A probe JSON that cannot be read or does not match is ignored (with the
    mismatch reported in ``source``), never fatal — the table is the
    fallback, not the probe.
    """
    import json
    from pathlib import Path

    def _clamped(batch: int | None, source: str) -> tuple[int | None, str]:
        if (
            backend is not None
            and backend != "gpu"
            and isinstance(batch, int)
            and batch > _NON_GPU_BATCH_CAP
        ):
            return _NON_GPU_BATCH_CAP, f"{source}, clamped to {_NON_GPU_BATCH_CAP} on {backend}"
        return batch, source

    if output_dir is not None:
        suffix = "_sparse" if path == "sparse" else ""
        probe_path = Path(output_dir) / f"vmap_probe_{model}{suffix}.json"
        if probe_path.exists():
            try:
                data = json.loads(probe_path.read_text())
            except (OSError, ValueError) as exc:
                return _clamped(
                    vmap_batch_for(dataset, model, instrument, path=path),
                    f"table (probe JSON unreadable: {exc})",
                )
            probe_backend = data.get("backend")
            matches = (
                data.get("dataset") == dataset
                and data.get("model") == model
                and data.get("instrument") == instrument
                # A probe measured on another backend is not evidence for
                # this one (CPU memory analysis says nothing about A100 and
                # vice versa). Probes predating the backend field pass.
                and (probe_backend is None or backend is None or probe_backend == backend)
            )
            recommended = data.get("recommended_batch_size")
            if matches and isinstance(recommended, int) and recommended >= 1:
                return _clamped(recommended, f"probe ({probe_path.name})")
            return _clamped(
                vmap_batch_for(dataset, model, instrument, path=path),
                f"table (probe JSON ignored: "
                f"{'cell mismatch' if not matches else 'no valid recommendation'})",
            )
    return _clamped(vmap_batch_for(dataset, model, instrument, path=path), "table")
