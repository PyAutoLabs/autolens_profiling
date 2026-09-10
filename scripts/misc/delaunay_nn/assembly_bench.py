"""Benchmark alternative formulations of the DelaunayNN ConstantSplit
regularization-matrix assembly (PyAutoArray ``reg_split_from`` +
``pixel_splitted_regularization_matrix_from``).

Diagnostic only: nothing here is imported by the library. Inputs are the real
(4P, K) split stencil tables built once from the HST DelaunayNN cell's
source-plane mesh (``build_tables.py``), loaded from
``results/delaunay_nn/tables_hst_1500.npz`` by default.

``V0`` is the pre-#536 assembly (the full padded ``(4P, K, K)`` scatter, forced
by ``compact_width=K``) and is the reference every other variant is checked
against; ``V6_library`` is the shipped library path with its default compaction
(``SPLIT_REG_COMPACT_WIDTH`` + the wide-row supplement), so the bench measures
what production actually runs.

Every variant is timed as a device-argument ``jax.jit`` function (unbatched)
and as ``jax.jit(jax.vmap(...))`` over a batch of 16, fp64, with
``block_until_ready`` and 3 warm-ups + median of N repeats.

Run from the autolens_profiling root:

```bash
python scripts/misc/delaunay_nn/assembly_bench.py
```
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import numpy as np


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
DEFAULT_TABLES = ROOT / "results" / "delaunay_nn" / "tables_hst_1500.npz"
DEFAULT_OUT = ROOT / "results" / "delaunay_nn" / "assembly_bench.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--tables", default=str(DEFAULT_TABLES), help=".npz from build_tables.py")
parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON output path")
parser.add_argument(
    "--pixels", type=int, default=0, help="truncate to this many mesh pixels (smoke)"
)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--skip", default="", help="comma-separated variant names to skip")
parser.add_argument("--label", default="", help="free-text label recorded in the JSON")
parser.add_argument(
    "--compact-widths",
    default="12,16",
    help="comma-separated fixed compaction widths for the V5 diagnostic (no supplement)",
)
args = parser.parse_args()

SKIP = {s.strip() for s in args.skip.split(",") if s.strip()}

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from autoarray.inversion.regularization import regularization_util as reg_util

BACKEND = jax.default_backend()
print(f"backend        : {BACKEND}")
print(f"devices        : {jax.devices()}")
print(f"x64 enabled    : {jax.config.jax_enable_x64}")

try:
    import autoarray

    print(f"autoarray      : {autoarray.__file__}")
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

data = np.load(args.tables)
mappings_np = data["mappings"].astype(np.int32)  # (S, K)
sizes_np = data["sizes"].astype(np.int32)  # (S,)
weights_np = data["weights"].astype(np.float64)  # (S, K)

if args.pixels:
    keep_pixels = int(args.pixels)
    keep_rows = 4 * keep_pixels
    mappings_np = mappings_np[:keep_rows].copy()
    sizes_np = sizes_np[:keep_rows].copy()
    weights_np = weights_np[:keep_rows].copy()
    # Drop stencil entries that point at dropped pixels; renormalise the row.
    bad = mappings_np >= keep_pixels
    mappings_np = np.where(bad, -1, mappings_np)
    weights_np = np.where(bad, 0.0, weights_np)
    # Compact each row so the surviving entries are contiguous within size.
    for r in range(mappings_np.shape[0]):
        k = int(sizes_np[r])
        m = mappings_np[r, :k]
        w = weights_np[r, :k]
        good = m >= 0
        m2, w2 = m[good], w[good]
        n = m2.shape[0]
        mappings_np[r, :] = -1
        weights_np[r, :] = 0.0
        mappings_np[r, :n] = m2
        weights_np[r, :n] = w2 / (w2.sum() if w2.sum() != 0 else 1.0)
        sizes_np[r] = n

S, K = mappings_np.shape
P = S // 4
BATCH = args.batch
REPEATS = args.repeats

print(f"tables         : S={S} K={K} P={P}")
print(f"sizes          : min={sizes_np.min()} max={sizes_np.max()} mean={sizes_np.mean():.2f}")


def duplicate_stats(mappings, sizes):
    """Unique (i, j) pairs vs total contributions in the (P, P) scatter."""
    valid = mappings != -1
    rows_i = []
    rows_j = []
    total = 0
    for r in range(mappings.shape[0]):
        m = mappings[r][valid[r]]
        total += m.size * m.size
        ii, jj = np.meshgrid(m, m, indexing="ij")
        rows_i.append(ii.ravel())
        rows_j.append(jj.ravel())
    ii = np.concatenate(rows_i)
    jj = np.concatenate(rows_j)
    keys = ii.astype(np.int64) * P + jj.astype(np.int64)
    uniq, counts = np.unique(keys, return_counts=True)
    return {
        "padded_scatter_entries": int(S * K * K),
        "real_contributions": int(total),
        "unique_ij_pairs": int(uniq.size),
        "max_multiplicity": int(counts.max()),
        "mean_multiplicity": float(counts.mean()),
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "size_mean": float(sizes.mean()),
    }


DUP = duplicate_stats(mappings_np, sizes_np)
print("duplicate stats:", DUP)


# ---------------------------------------------------------------------------
# Device inputs
# ---------------------------------------------------------------------------

mappings = jnp.asarray(mappings_np)
sizes = jnp.asarray(sizes_np)
weights = jnp.asarray(weights_np)
reg_w = jnp.full((P,), 1.0, dtype=jnp.float64)  # ConstantSplit(coefficient=1.0)

# Batch: mappings/sizes are the same connectivity, weights perturbed per lane
# (under a production vmap every lane re-runs Sibson, so all four are batched).
rng = np.random.default_rng(0)
pert = 1.0 + 1e-3 * rng.standard_normal((BATCH, 1, 1))
weights_b = jnp.asarray(weights_np[None, :, :] * pert)
mappings_b = jnp.broadcast_to(mappings, (BATCH,) + mappings.shape)
sizes_b = jnp.broadcast_to(sizes, (BATCH,) + sizes.shape)
reg_w_b = jnp.broadcast_to(reg_w, (BATCH,) + reg_w.shape)
mappings_b = jnp.asarray(mappings_b)
sizes_b = jnp.asarray(sizes_b)
reg_w_b = jnp.asarray(reg_w_b)

# 4-wide (Delaunay-style) calibration tables: 3 real columns + 1 spare.
K4 = 4
mappings4_np = -np.ones((S, K4), dtype=np.int32)
weights4_np = np.zeros((S, K4), dtype=np.float64)
sizes4_np = np.minimum(sizes_np, 3).astype(np.int32)
for r in range(S):
    k = int(sizes4_np[r])
    mappings4_np[r, :k] = mappings_np[r, :k]
    w = weights_np[r, :k]
    tot = w.sum()
    weights4_np[r, :k] = w / (tot if tot != 0 else 1.0)
mappings4 = jnp.asarray(mappings4_np)
sizes4 = jnp.asarray(sizes4_np)
weights4 = jnp.asarray(weights4_np)
mappings4_b = jnp.asarray(np.broadcast_to(mappings4_np, (BATCH, S, K4)))
sizes4_b = jnp.asarray(np.broadcast_to(sizes4_np, (BATCH, S)))
weights4_b = jnp.asarray(weights4_np[None, :, :] * pert)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def _split(mappings, sizes, weights):
    """``reg_split_from`` JAX branch — shared prologue of every variant."""
    return reg_util.reg_split_from(
        splitted_mappings=mappings,
        splitted_sizes=sizes,
        splitted_weights=weights,
        xp=jnp,
    )


def v0(mappings, sizes, weights, reg_weights):
    """Pre-#536 path: reg_split_from + the full padded (4P, K, K) scatter.

    ``compact_width=K`` disables the library's compaction, so this is the
    uncompacted scatter the compaction replaced, and the reference result.
    """
    m, s, w = _split(mappings, sizes, weights)
    return reg_util.pixel_splitted_regularization_matrix_from(
        regularization_weights=reg_weights,
        splitted_mappings=m,
        splitted_sizes=s,
        splitted_weights=w,
        xp=jnp,
        compact_width=m.shape[1],
    )


def v6(mappings, sizes, weights, reg_weights):
    """The shipped library path: default compaction + wide-row supplement."""
    m, s, w = _split(mappings, sizes, weights)
    return reg_util.pixel_splitted_regularization_matrix_from(
        regularization_weights=reg_weights,
        splitted_mappings=m,
        splitted_sizes=s,
        splitted_weights=w,
        xp=jnp,
    )


def _dense_b(m, w, reg_weights):
    """Dense split-mapping matrix B (S, P) and the per-row scale s (S,)."""
    n_rows = m.shape[0]
    n_pix = reg_weights.shape[0]
    valid = m != -1
    m_fixed = jnp.where(valid, m, 0)
    w_fixed = jnp.where(valid, w, 0.0)
    rows = jnp.arange(n_rows, dtype=m_fixed.dtype)[:, None]
    rows = jnp.broadcast_to(rows, m_fixed.shape)
    b = jnp.zeros((n_rows, n_pix), dtype=w.dtype).at[rows, m_fixed].add(w_fixed)
    scale = reg_weights[jnp.arange(n_rows) // 4] ** 2.0
    return b, scale


def _v1(mappings, sizes, weights, reg_weights, precision=None):
    m, s, w = _split(mappings, sizes, weights)
    b, scale = _dense_b(m, w, reg_weights)
    h = jnp.matmul(b.T, scale[:, None] * b, precision=precision)
    return h + jnp.eye(reg_weights.shape[0], dtype=w.dtype) * 1e-8


def v1(mappings, sizes, weights, reg_weights):
    """Dense GEMM: H = B^T diag(s) B."""
    return _v1(mappings, sizes, weights, reg_weights)


def v1_highest(mappings, sizes, weights, reg_weights):
    return _v1(mappings, sizes, weights, reg_weights, precision=jax.lax.Precision.HIGHEST)


def v2(mappings, sizes, weights, reg_weights):
    """BCOO sparse B^T @ dense (scaled) B."""
    from jax.experimental import sparse

    m, s, w = _split(mappings, sizes, weights)
    n_rows = m.shape[0]
    n_pix = reg_weights.shape[0]
    valid = m != -1
    m_fixed = jnp.where(valid, m, 0)
    w_fixed = jnp.where(valid, w, 0.0)
    rows = jnp.broadcast_to(jnp.arange(n_rows, dtype=m_fixed.dtype)[:, None], m_fixed.shape)
    idx = jnp.stack([rows.ravel(), m_fixed.ravel()], axis=-1)
    scale = reg_weights[jnp.arange(n_rows) // 4] ** 2.0
    b_sp = sparse.BCOO((w_fixed.ravel(), idx), shape=(n_rows, n_pix))
    b_dense = jnp.zeros((n_rows, n_pix), dtype=w.dtype).at[rows, m_fixed].add(w_fixed)
    h = b_sp.T @ (scale[:, None] * b_dense)
    return h + jnp.eye(n_pix, dtype=w.dtype) * 1e-8


def v2b(mappings, sizes, weights, reg_weights):
    """BCOO sparse-sparse: (sqrt(s) B)^T @ (sqrt(s) B) via bcoo_dot_general."""
    from jax.experimental import sparse

    m, s, w = _split(mappings, sizes, weights)
    n_rows = m.shape[0]
    n_pix = reg_weights.shape[0]
    valid = m != -1
    m_fixed = jnp.where(valid, m, 0)
    w_fixed = jnp.where(valid, w, 0.0)
    scale = jnp.sqrt(reg_weights[jnp.arange(n_rows) // 4] ** 2.0)
    w_scaled = w_fixed * scale[:, None]
    rows = jnp.broadcast_to(jnp.arange(n_rows, dtype=m_fixed.dtype)[:, None], m_fixed.shape)
    idx_t = jnp.stack([m_fixed.ravel(), rows.ravel()], axis=-1)  # B^T indices
    bt_sp = sparse.BCOO((w_scaled.ravel(), idx_t), shape=(n_pix, n_rows))
    idx = jnp.stack([rows.ravel(), m_fixed.ravel()], axis=-1)
    b_sp = sparse.BCOO((w_scaled.ravel(), idx), shape=(n_rows, n_pix))
    h = sparse.bcoo_dot_general(
        bt_sp,
        b_sp,
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    h = h.todense() if hasattr(h, "todense") else h
    return h + jnp.eye(n_pix, dtype=w.dtype) * 1e-8


def v3(mappings, sizes, weights, reg_weights):
    """Dedup-then-scatter: sort the (i, j) triples, segment_sum, one scatter."""
    m, s, w = _split(mappings, sizes, weights)
    n_rows = m.shape[0]
    n_pix = reg_weights.shape[0]
    valid = m != -1
    m_fixed = jnp.where(valid, m, 0)
    w_fixed = jnp.where(valid, w, 0.0)
    scale = reg_weights[jnp.arange(n_rows) // 4] ** 2.0
    outer = (w_fixed[:, :, None] * w_fixed[:, None, :]) * scale[:, None, None]
    ii = jnp.broadcast_to(m_fixed[:, :, None], outer.shape)
    jj = jnp.broadcast_to(m_fixed[:, None, :], outer.shape)
    keys = (ii.astype(jnp.int32) * n_pix + jj.astype(jnp.int32)).ravel()
    vals = outer.ravel()
    order = jnp.argsort(keys)
    keys_s = keys[order]
    vals_s = vals[order]
    new = jnp.concatenate(
        [jnp.ones((1,), dtype=jnp.int32), (keys_s[1:] != keys_s[:-1]).astype(jnp.int32)]
    )
    seg = jnp.cumsum(new) - 1
    n_seg = keys_s.shape[0]
    summed = jax.ops.segment_sum(vals_s, seg, num_segments=n_seg, indices_are_sorted=True)
    # First key of each segment.
    seg_keys = jnp.zeros((n_seg,), dtype=keys_s.dtype).at[seg].max(keys_s)
    flat = jnp.zeros((n_pix * n_pix,), dtype=w.dtype).at[seg_keys].add(summed)
    return flat.reshape((n_pix, n_pix)) + jnp.eye(n_pix, dtype=w.dtype) * 1e-8


def make_v5(kc: int):
    """Fixed stencil compaction to ``kc`` columns, with no wide-row supplement.

    Numerically identical to V0 whenever every post-split size <= kc, which the
    real HST tables satisfy (max size 11 + 1 insertion). Kept as the width sweep
    that priced the compaction: it isolates the cost of the main scatter alone,
    where ``V6_library`` adds the supplement that makes the result exact in the
    tail.
    """

    def v5(mappings, sizes, weights, reg_weights):
        m, s, w = _split(mappings, sizes, weights)
        m = m[:, :kc]
        w = w[:, :kc]
        return reg_util.pixel_splitted_regularization_matrix_from(
            regularization_weights=reg_weights,
            splitted_mappings=m,
            splitted_sizes=jnp.minimum(s, kc),
            splitted_weights=w,
            xp=jnp,
            compact_width=kc,
        )

    return v5


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


def _block(x):
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def peak_mem_bytes():
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats is None:
            return None
        return int(stats.get("peak_bytes_in_use", 0))
    except Exception:
        return None


def time_fn(fn, arrays, repeats=REPEATS, warmups=3):
    compiled = jax.jit(fn)
    result = _block(compiled(*arrays))
    for _ in range(warmups - 1):
        _block(compiled(*arrays))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _block(compiled(*arrays))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3, result


def run_variant(name, fn, single_args, batch_args, check_ref=None, batched=True):
    entry = {"variant": name, "backend": BACKEND}
    try:
        ms, result = time_fn(fn, single_args)
        entry["unbatched_ms"] = ms
        entry["peak_mem_mib_after_unbatched"] = (
            (peak_mem_bytes() or 0) / 2**20 if peak_mem_bytes() is not None else None
        )
        if check_ref is not None:
            ref = np.asarray(check_ref, dtype=np.float64)
            got = np.asarray(result, dtype=np.float64)
            diff = np.abs(got - ref)
            denom = np.abs(ref)
            entry["max_abs_diff"] = float(diff.max())
            nz = denom > 0
            entry["max_rel_diff"] = float((diff[nz] / denom[nz]).max()) if nz.any() else 0.0
        print(f"  {name:<16} unbatched {ms:9.3f} ms", flush=True)
    except Exception as exc:  # pragma: no cover
        entry["unbatched_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  {name:<16} unbatched FAILED: {exc}", flush=True)
        traceback.print_exc()
        result = None

    if batched:
        try:
            ms_b, _ = time_fn(jax.vmap(fn), batch_args)
            entry["batched_total_ms"] = ms_b
            entry["per_call_vmap_ms"] = ms_b / BATCH
            entry["peak_mem_mib_after_batched"] = (
                (peak_mem_bytes() or 0) / 2**20 if peak_mem_bytes() is not None else None
            )
            print(f"  {name:<16} vmap{BATCH:<3d}   {ms_b / BATCH:9.3f} ms/call", flush=True)
        except Exception as exc:  # pragma: no cover
            entry["batched_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {name:<16} vmap{BATCH:<3d}   FAILED: {exc}", flush=True)
    return entry, result


single_args = (mappings, sizes, weights, reg_w)
batch_args = (mappings_b, sizes_b, weights_b, reg_w_b)
single_args4 = (mappings4, sizes4, weights4, reg_w)
batch_args4 = (mappings4_b, sizes4_b, weights4_b, reg_w_b)

results = []

print("\n--- V0 (baseline) ---", flush=True)
entry, h_ref = run_variant("V0", v0, single_args, batch_args)
results.append(entry)
h_ref_np = np.asarray(h_ref, dtype=np.float64) if h_ref is not None else None
if h_ref_np is not None:
    print(
        f"  H: shape {h_ref_np.shape} nnz {int((h_ref_np != 0).sum())} "
        f"frobenius {np.linalg.norm(h_ref_np):.6e}"
    )
    sign, logdet = np.linalg.slogdet(h_ref_np)
    print(f"  slogdet(H) = ({sign}, {logdet:.9f})")
    results[-1]["h_slogdet"] = [float(sign), float(logdet)]
    results[-1]["h_frobenius"] = float(np.linalg.norm(h_ref_np))

VARIANTS = [
    ("V0a_4wide", v0, single_args4, batch_args4, None),
    ("V1_dense_gemm", v1, single_args, batch_args, h_ref_np),
    ("V1_highest", v1_highest, single_args, batch_args, h_ref_np),
    ("V2_bcoo_dense", v2, single_args, batch_args, h_ref_np),
    ("V2b_bcoo_bcoo", v2b, single_args, batch_args, h_ref_np),
    ("V3_dedup_segsum", v3, single_args, batch_args, h_ref_np),
]

for _kc in [int(x) for x in args.compact_widths.split(",") if x.strip()]:
    VARIANTS.append((f"V5_compact_k{_kc}", make_v5(_kc), single_args, batch_args, h_ref_np))

VARIANTS.append(("V6_library", v6, single_args, batch_args, h_ref_np))

for name, fn, sa, ba, ref in VARIANTS:
    if name in SKIP:
        print(f"\n--- {name} (skipped) ---", flush=True)
        results.append({"variant": name, "backend": BACKEND, "skipped": True})
        continue
    print(f"\n--- {name} ---", flush=True)
    entry, res = run_variant(name, fn, sa, ba, check_ref=ref)
    if name == "V1_dense_gemm" and res is not None and h_ref_np is not None:
        try:
            s1, l1 = np.linalg.slogdet(np.asarray(res, dtype=np.float64))
            entry["h_slogdet"] = [float(s1), float(l1)]
        except Exception:
            pass
    results.append(entry)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

header = (
    f"{'variant':<18}{'backend':<9}{'unbatched ms':>14}{'per-call@vmap' + str(BATCH):>18}"
    f"{'max abs diff':>16}{'max rel diff':>16}{'peak MiB':>12}"
)
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
for r in results:
    if r.get("skipped"):
        print(f"{r['variant']:<18}{r['backend']:<9}{'skipped':>14}")
        continue
    ub = r.get("unbatched_ms")
    pc = r.get("per_call_vmap_ms")
    ad = r.get("max_abs_diff")
    rd = r.get("max_rel_diff")
    pm = r.get("peak_mem_mib_after_batched") or r.get("peak_mem_mib_after_unbatched")
    print(
        f"{r['variant']:<18}{r['backend']:<9}"
        f"{(f'{ub:.3f}' if ub is not None else 'FAIL'):>14}"
        f"{(f'{pc:.3f}' if pc is not None else 'FAIL'):>18}"
        f"{(f'{ad:.3e}' if ad is not None else '-'):>16}"
        f"{(f'{rd:.3e}' if rd is not None else '-'):>16}"
        f"{(f'{pm:.0f}' if pm is not None else '-'):>12}"
    )
print("=" * len(header))

payload = {
    "label": args.label,
    "backend": BACKEND,
    "devices": [str(d) for d in jax.devices()],
    "x64": bool(jax.config.jax_enable_x64),
    "S": int(S),
    "K": int(K),
    "P": int(P),
    "batch": BATCH,
    "repeats": REPEATS,
    "tables": os.path.abspath(args.tables),
    "duplicate_stats": DUP,
    "xla_flags": os.environ.get("XLA_FLAGS"),
    "results": results,
}
try:
    import autoarray

    payload["autoarray_file"] = autoarray.__file__
    payload["autoarray_version"] = getattr(autoarray, "__version__", None)
except Exception:
    pass

with open(args.out, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nwrote {args.out}")
