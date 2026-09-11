"""Aggregator for the reduced N_src size sweep — autolens_profiling#247, phase 3.

Renders the 26-leg sweep (plus the fiducial n=1500 rows) as three markdown
sections: which paths still **fit**, what one call **costs**, and where — if
anywhere — the matrix-free line **crosses** the exact one.

Run from the repo root::

    python scripts/misc/likelihood_breakdown/size_sweep_table.py
    python scripts/misc/likelihood_breakdown/size_sweep_table.py --out /tmp/sweep.md

Why a leg with no JSON is still a row
-------------------------------------

After the CPU reference measured ``cond(F + λH) ≈ 4e10`` the sweep's primary
question stopped being speed and became memory: **at what mesh size does the
exact path stop fitting an A100?** The answer to that question is a leg that
died — so a leg that produced no result JSON is not a gap in the table, it is
the measurement. This script therefore reads three sources, in order:

1. ``results/sweep/size_sweep_manifest.jsonl`` — one line per leg the launcher
   submitted (``job``, ``mesh``, ``path``, ``n``, ``node``, ``submitted_utc``,
   ``sha``). This is the only record that distinguishes *submitted and died*
   from *never submitted*. Optional: without it the script falls back to the
   ``SWEEP_LEG`` headers in the SLURM ``.out`` logs, and then to whatever result
   JSONs it can glob.
2. ``results/breakdown/imaging/*.json`` — the rows themselves.
3. ``hpc/batch_gpu/{output,error}/`` — the logs of a submitted leg with no JSON,
   classified as ``oom`` / ``timeout`` / ``failed`` / ``pending`` / ``missing``.

Nothing here is interpolated or extrapolated. A cell with no measurement prints
``n/a — measured, not extrapolated``; a crossover that has not been measured is
reported as not measured, never as a fitted curve.

Filenames it reads
------------------

``resolve_output_paths`` builds a config-tagged basename as
``<cell>_<config_name>`` and then appends ``_sparse`` when ``--sparse`` is set,
so the swept legs land as:

===========  ====================================================
path         basename
===========  ====================================================
dense        ``<cell>_hpc_a100_fp64_n<N>``
sparse       ``<cell>_hpc_a100_fp64_n<N>_sparse``
matrix_free  ``matrix_free_<mesh>_hpc_a100_fp64_matrix_free_n<N>``
===========  ====================================================

with ``<cell>`` one of ``pixelization`` / ``delaunay`` / ``delaunay_nn`` and the
matrix-free cell naming itself ``matrix_free_<mesh>`` (so the three meshes'
rows stay disjoint). At the fiducial n=1500 the basenames carry no ``_n``
suffix, and the **dense** row is read from the ``_recon_split`` leg in
preference to the plain one: only the recon-split leg carries the exact
``Cholesky solve (unconstrained)`` and ``Log det Cholesky`` comparators the
crossover is measured against.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()

#: The fiducial tier every sweep row is anchored to (39x39 = 1521 rectangular
#: pixels / 1500 Delaunay vertices). Its rows already exist and are never
#: re-run by the sweep.
FIDUCIAL_N = 1500

MESHES = ("rectangular", "delaunay", "delaunay_nn")
PATHS = ("dense", "sparse", "matrix_free")

#: Which breakdown cell each mesh's dense / sparse legs run.
CELL_FOR_MESH = {
    "rectangular": "pixelization",
    "delaunay": "delaunay",
    "delaunay_nn": "delaunay_nn",
}

#: The SLQ pair the sweep's matrix-free legs make their default: p=16, and
#: m=320 because ``--slq-steps 20,80,320`` makes 320 the last (and so the
#: cell's ``SLQ_DEFAULT_STEPS``). ``matrix_free.evidence.log_evidence_error``
#: is the error of exactly this pair.
SLQ_PROBES = 16
SLQ_STEPS = 320

#: Exact-path comparator rows, in ``steps_reconstruction_sub_rows``. Written
#: unconditionally by the three breakdown cells since the 2026-09-11
#: reconstruction split, so every swept dense/sparse leg carries them — but the
#: pre-split fiducial ``hpc_a100_fp64`` rows do not, which is why the fiducial
#: dense row is read from the ``_recon_split`` leg.
EXACT_SOLVE_ROW = "Cholesky solve (unconstrained)"
EXACT_LOGDET_ROWS = ("Log det Cholesky (F+λH reduced)", "Log det Cholesky (H reduced)")
EXACT_VMAP_ROW_RE = re.compile(r"^NNLS PDIP @vmap \d+ \(identical lanes\)$")

#: Matrix-free rows. ``steps_matrix_free_rows`` is a list of ``{name, ms, ...}``.
MF_SOLVE_ROW = "PCG solve (unconstrained, Jacobi est.)"
MF_SLQ_CURV_ROW = f"SLQ log det (F+λH reduced) p={SLQ_PROBES} m={SLQ_STEPS}"
MF_SLQ_REG_ROW = f"SLQ log det (λH reduced) p={SLQ_PROBES} m={SLQ_STEPS}"
MF_VMAP_ROW_RE = re.compile(r"^PCG solve @vmap \d+ \(identical lanes\)$")

#: Device peak-memory keys, in preference order (dotted paths into the JSON).
#: The matrix-free cell records three high-water marks; the dense/sparse cells
#: record none today, so their fits cell is a bare tick.
PEAK_BYTES_KEYS = (
    "matrix_free.peak_bytes.after_matrix_free",
    "matrix_free.peak_bytes.after_dense_comparators",
    "matrix_free.peak_bytes.after_setup",
    "dense.peak_bytes",
    "device.peak_bytes",
    "device.peak_bytes_in_use",
    "peak_bytes",
)

#: Log signatures. Checked in this order: an out-of-memory death answers the
#: sweep's primary question, so it wins over the SLURM time-limit notice and
#: over a traceback that the OOM itself produced.
OOM_SIGNATURES = (
    "RESOURCE_EXHAUSTED",
    "Out of memory",
    "out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
)
TIMEOUT_SIGNATURES = ("DUE TO TIME LIMIT",)
FAILED_SIGNATURES = ("Traceback",)

#: ``SWEEP_LEG mesh=<mesh> path=<path> n=<N> job=<id>`` — the first line every
#: leg prints, so a ``.out`` maps back to its leg with no manifest.
SWEEP_LEG_RE = re.compile(
    r"SWEEP_LEG\s+mesh=(?P<mesh>\S+)\s+path=(?P<path>\S+)\s+n=(?P<n>\d+)\s+job=(?P<job>\S+)"
)

#: What the crossover paragraph prints for an absent leg. The tables use the
#: short form ``n/a`` with the same meaning, spelled out in their legend: a
#: missing cell is a leg that was not measured, and nothing in this file is
#: ever filled in by interpolation or a fitted curve.
NOT_MEASURED = "n/a — measured, not extrapolated"
NA = "n/a"


# ---------------------------------------------------------------------------
# leg identity and file resolution
# ---------------------------------------------------------------------------


def json_candidates(mesh: str, path: str, n: int) -> list[str]:
    """Result-JSON basenames for one leg, most-preferred first."""
    cell = CELL_FOR_MESH.get(mesh, mesh)
    if n == FIDUCIAL_N:
        if path == "dense":
            # The recon-split leg first: it is the only fiducial dense row that
            # carries the exact solve / log-det comparators.
            return [f"{cell}_hpc_a100_fp64_recon_split", f"{cell}_hpc_a100_fp64"]
        if path == "sparse":
            return [f"{cell}_hpc_a100_fp64_recon_split_sparse", f"{cell}_hpc_a100_fp64_sparse"]
        return [f"matrix_free_{mesh}_hpc_a100_fp64_matrix_free"]
    if path == "dense":
        return [f"{cell}_hpc_a100_fp64_n{n}"]
    if path == "sparse":
        return [f"{cell}_hpc_a100_fp64_n{n}_sparse"]
    return [f"matrix_free_{mesh}_hpc_a100_fp64_matrix_free_n{n}"]


def find_json(results_dir: Path, mesh: str, path: str, n: int) -> Path | None:
    for basename in json_candidates(mesh, path, n):
        candidate = results_dir / f"{basename}.json"
        if candidate.exists():
            return candidate
    return None


def glob_legs(results_dir: Path) -> set[tuple[str, str, int]]:
    """Every ``(mesh, path, n)`` a result JSON in ``results_dir`` implies.

    The fallback when there is no manifest and no logs: it can only see legs
    that finished, which is precisely why the manifest exists.
    """
    legs: set[tuple[str, str, int]] = set()
    for json_path in sorted(results_dir.glob("*.json")):
        name = json_path.stem
        match = re.match(
            r"^matrix_free_(?P<mesh>\w+?)_hpc_a100_fp64_matrix_free(?:_n(\d+))?$", name
        )
        if match:
            mesh = match.group("mesh")
            if mesh in MESHES:
                legs.add((mesh, "matrix_free", int(match.group(2) or FIDUCIAL_N)))
            continue
        match = re.match(r"^(?P<cell>\w+?)_hpc_a100_fp64_n(?P<n>\d+)(?P<sparse>_sparse)?$", name)
        if not match:
            continue
        for mesh, cell in CELL_FOR_MESH.items():
            if match.group("cell") == cell:
                legs.add(
                    (mesh, "sparse" if match.group("sparse") else "dense", int(match.group("n")))
                )
    return legs


# ---------------------------------------------------------------------------
# manifest and logs
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> list[dict]:
    """The submitted legs, one dict per line. Missing file -> empty list."""
    if not manifest_path.exists():
        return []
    entries: list[dict] = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _read_capped(path: Path, limit: int = 262_144) -> str:
    """Head + tail of a log file. A SLURM ``.out`` can be large; the leg header
    is at the very top and the failure signature at the very bottom."""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > 2 * limit:
        data = data[:limit] + b"\n...\n" + data[-limit:]
    return data.decode("utf-8", errors="replace")


def scan_logs(logs_dir: Path) -> dict[tuple[str, str, int], str]:
    """``(mesh, path, n) -> job id``, read from the ``SWEEP_LEG`` header lines.

    Lets the table classify a leg whose manifest line is missing (a submit made
    by hand, or a manifest not yet committed back from RAL).
    """
    found: dict[tuple[str, str, int], str] = {}
    out_dir = logs_dir / "output"
    if not out_dir.is_dir():
        return found
    for out_path in sorted(out_dir.glob("output.*.out")):
        match = SWEEP_LEG_RE.search(_read_capped(out_path, 4096))
        if not match:
            continue
        found[(match["mesh"], match["path"], int(match["n"]))] = match["job"]
    return found


def classify_logs(logs_dir: Path, job: str | None) -> str:
    """Status of a submitted leg that produced no result JSON."""
    if job is None:
        return "pending"
    texts = []
    for sub, pattern in (("error", "error.{job}.err"), ("output", "output.{job}.out")):
        log_path = logs_dir / sub / pattern.format(job=job)
        if log_path.exists():
            texts.append(_read_capped(log_path))
    if not texts:
        return "pending"
    blob = "\n".join(texts)
    if any(sig in blob for sig in OOM_SIGNATURES):
        return "oom"
    if any(sig in blob for sig in TIMEOUT_SIGNATURES):
        return "timeout"
    if any(sig in blob for sig in FAILED_SIGNATURES):
        return "failed"
    return "missing"


# ---------------------------------------------------------------------------
# reading one leg
# ---------------------------------------------------------------------------


def _dig(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def peak_gb(payload: dict) -> float | None:
    for key in PEAK_BYTES_KEYS:
        value = _dig(payload, key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value) / 1024**3
    return None


def _row_by_name(rows: list, name: str) -> dict | None:
    for row in rows or []:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def _row_by_re(rows: list, pattern: re.Pattern) -> dict | None:
    for row in rows or []:
        if isinstance(row, dict) and pattern.match(str(row.get("name", ""))):
            return row
    return None


class Leg:
    """One (mesh, path, n) cell of the sweep, measured or not."""

    def __init__(self, mesh: str, path: str, n: int):
        self.mesh = mesh
        self.path = path
        self.n = n
        self.status = "pending"
        self.job: str | None = None
        self.json_path: Path | None = None
        self.payload: dict = {}

    # -- fits ------------------------------------------------------------
    @property
    def fits(self) -> bool:
        return self.status == "ok"

    def fits_cell(self) -> str:
        if self.status != "ok":
            return {"oom": "OOM", "timeout": "timeout"}.get(self.status, self.status)
        gb = peak_gb(self.payload)
        return "✓" if gb is None else f"✓ {gb:.1f} GB"

    # -- source pixels actually built -------------------------------------
    @property
    def built_pixels(self) -> int | None:
        conf = self.payload.get("configuration", {})
        for key in ("source_pixels", "delaunay_vertices"):
            if isinstance(conf.get(key), int):
                return conf[key]
        return None

    # -- per-call --------------------------------------------------------
    @property
    def mf_rows(self) -> list:
        return self.payload.get("steps_matrix_free_rows", []) or []

    @property
    def sub_rows(self) -> dict:
        return self.payload.get("steps_reconstruction_sub_rows", {}) or {}

    def per_call_ms(self) -> float | None:
        """Unbatched per-call cost of this path, in ms.

        Exact paths: the cell's step-sum total (``total_step_by_step``) — the
        whole likelihood. Matrix-free: the three rows the crossover turns on
        (the PCG solve and the two SLQ log-dets), which is what replaces the
        exact path's solve + two log-det Choleskys, NOT a whole likelihood.
        The two columns are therefore not comparable to each other; the
        comparison the crossover makes is `components`, below.
        """
        if self.status != "ok":
            return None
        if self.path == "matrix_free":
            parts = self.mf_components()
            if any(v is None for v in parts.values()) or not parts:
                return None
            return sum(parts.values())
        total = self.payload.get("total_step_by_step")
        return float(total) * 1e3 if isinstance(total, (int, float)) else None

    def mf_components(self) -> dict[str, float | None]:
        """``pcg`` / ``slq_curv`` / ``slq_reg`` per-call ms."""
        out: dict[str, float | None] = {}
        for key, name in (
            ("pcg", MF_SOLVE_ROW),
            ("slq_curv", MF_SLQ_CURV_ROW),
            ("slq_reg", MF_SLQ_REG_ROW),
        ):
            row = _row_by_name(self.mf_rows, name)
            out[key] = float(row["ms"]) if row and row.get("ms") is not None else None
        return out

    def exact_components(self) -> dict[str, float | None]:
        """``solve`` / ``logdet_curv`` / ``logdet_reg`` per-call ms.

        The exact comparator the matrix-free line has to beat: one Cholesky
        solve plus the two log-det Choleskys, on the same model at the same N.
        """
        rows = self.sub_rows
        return {
            "solve": rows.get(EXACT_SOLVE_ROW),
            "logdet_curv": rows.get(EXACT_LOGDET_ROWS[0]),
            "logdet_reg": rows.get(EXACT_LOGDET_ROWS[1]),
        }

    def exact_total_ms(self) -> float | None:
        parts = self.exact_components()
        if any(v is None for v in parts.values()):
            return None
        return sum(float(v) * 1e3 for v in parts.values())

    def vmap_solve_ms(self) -> float | None:
        """The batched (``@vmap 16``) per-call of this path's solve row."""
        if self.status != "ok":
            return None
        if self.path == "matrix_free":
            row = _row_by_re(self.mf_rows, MF_VMAP_ROW_RE)
            return float(row["ms"]) if row and row.get("ms") is not None else None
        for name, value in self.sub_rows.items():
            if EXACT_VMAP_ROW_RE.match(name):
                return float(value) * 1e3
        return None

    def cg_iterations(self) -> int | None:
        row = _row_by_name(self.mf_rows, MF_SOLVE_ROW)
        if row and isinstance(row.get("cg_iterations"), int):
            return row["cg_iterations"]
        return None

    def cond(self) -> float | None:
        value = _dig(self.payload, "matrix_free.cond_curvature_reg")
        return float(value) if isinstance(value, (int, float)) else None

    def log_evidence_error(self) -> float | None:
        value = _dig(self.payload, "matrix_free.evidence.log_evidence_error")
        return float(value) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def collect(
    results_dir: Path,
    manifest_path: Path,
    logs_dir: Path,
) -> tuple[dict[tuple[str, str, int], Leg], list[dict]]:
    """Every leg the sweep knows about, keyed by ``(mesh, path, n)``."""
    manifest = load_manifest(manifest_path)
    log_jobs = scan_logs(logs_dir)

    expected: dict[tuple[str, str, int], str | None] = {}
    # The fiducial tier always has a row: it is the anchor the sweep is read
    # against, and its absence would silently drop the comparison.
    for mesh in MESHES:
        for path in PATHS:
            expected[(mesh, path, FIDUCIAL_N)] = None
    for entry in manifest:
        try:
            key = (str(entry["mesh"]), str(entry["path"]), int(entry["n"]))
        except (KeyError, TypeError, ValueError):
            continue
        expected[key] = str(entry.get("job")) if entry.get("job") is not None else None
    for key, job in log_jobs.items():
        expected.setdefault(key, job)
        if expected[key] is None:
            expected[key] = job
    for key in glob_legs(results_dir):
        expected.setdefault(key, None)

    legs: dict[tuple[str, str, int], Leg] = {}
    for (mesh, path, n), job in sorted(
        expected.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])
    ):
        leg = Leg(mesh, path, n)
        leg.job = job
        json_path = find_json(results_dir, mesh, path, n)
        if json_path is not None:
            try:
                leg.payload = json.loads(json_path.read_text())
                leg.json_path = json_path
                leg.status = "ok"
            except (OSError, json.JSONDecodeError):
                leg.status = "failed"
        else:
            leg.status = classify_logs(logs_dir, job)
        legs[(mesh, path, n)] = leg
    return legs, manifest


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _fmt_ms(value: float | None) -> str:
    return NA if value is None else f"{value:.3f}"


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return NA
    return format(value, spec) if spec else str(value)


def _sizes(legs: dict[tuple[str, str, int], Leg]) -> list[int]:
    return sorted({key[2] for key in legs})


def render_fits_table(legs: dict[tuple[str, str, int], Leg]) -> list[str]:
    lines = [
        "## 1. Fits — does this path still run at this mesh size?",
        "",
        "`✓` with a peak-device-memory figure where the leg records one (the matrix-free cell",
        "reports `matrix_free.peak_bytes`; the dense and sparse cells record none today, so their",
        "`✓` is bare). `OOM` is a **datum**, not a failure: it is the answer to the question the",
        "sweep was reduced to ask. `—` is a leg the sweep never submitted.",
        "",
        "| mesh | N (requested) | built | dense | sparse | matrix-free |",
        "|---|---|---|---|---|---|",
    ]
    for n in _sizes(legs):
        for mesh in MESHES:
            present = [legs.get((mesh, path, n)) for path in PATHS]
            if not any(present):
                continue
            built = next((leg.built_pixels for leg in present if leg and leg.built_pixels), None)
            cells = [leg.fits_cell() if leg else "—" for leg in present]
            lines.append(
                f"| {mesh} | {n} | {built if built is not None else '—'} | "
                + " | ".join(cells)
                + " |"
            )
    lines.append("")
    return lines


def render_per_call_table(legs: dict[tuple[str, str, int], Leg]) -> list[str]:
    lines = [
        "## 2. Per call — what one likelihood evaluation costs",
        "",
        "`per-call ms` is the **exact paths'** whole-likelihood step-sum total",
        "(`total_step_by_step`) and the **matrix-free** path's three crossover rows summed",
        f"(`{MF_SOLVE_ROW}` + `{MF_SLQ_CURV_ROW}` + `{MF_SLQ_REG_ROW}`). Those two are different",
        "quantities and must not be compared column-to-column — the like-for-like comparison is",
        "`components`, which on the exact paths is the solve plus the two log-det Choleskys that",
        "the matrix-free rows replace. `solve @vmap16` is the batched per-call of each path's own",
        "solve row (`NNLS PDIP @vmap N` / `PCG solve @vmap N`). `n/a` is a value this leg does",
        "not record; no cell is ever interpolated.",
        "",
        "| mesh | N | path | per-call ms | components (ms) | solve @vmap16 ms | cg iters | cond(F+λH) | log-ev err (nats) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n in _sizes(legs):
        for mesh in MESHES:
            for path in PATHS:
                leg = legs.get((mesh, path, n))
                if leg is None:
                    continue
                if leg.status != "ok":
                    lines.append(
                        f"| {mesh} | {n} | {path} | {leg.status} | {NA} | "
                        f"{NA} | {NA} | {NA} | {NA} |"
                    )
                    continue
                if path == "matrix_free":
                    parts = leg.mf_components()
                    components = (
                        f"pcg {_fmt_ms(parts['pcg'])} · slq(F+λH) {_fmt_ms(parts['slq_curv'])}"
                        f" · slq(λH) {_fmt_ms(parts['slq_reg'])}"
                    )
                else:
                    parts = leg.exact_components()
                    components = (
                        f"chol solve {_fmt_ms(None if parts['solve'] is None else parts['solve'] * 1e3)}"
                        f" · logdet(F+λH) {_fmt_ms(None if parts['logdet_curv'] is None else parts['logdet_curv'] * 1e3)}"
                        f" · logdet(H) {_fmt_ms(None if parts['logdet_reg'] is None else parts['logdet_reg'] * 1e3)}"
                    )
                lines.append(
                    f"| {mesh} | {n} | {path} | {_fmt_ms(leg.per_call_ms())} | {components} | "
                    f"{_fmt_ms(leg.vmap_solve_ms())} | {_fmt(leg.cg_iterations())} | "
                    f"{_fmt(leg.cond(), '.3g')} | {_fmt(leg.log_evidence_error(), '+.3f')} |"
                )
    lines.append("")
    return lines


def _first_not_fitting(legs, mesh: str, path: str) -> int | None:
    for n in _sizes(legs):
        leg = legs.get((mesh, path, n))
        if leg is not None and leg.status in ("oom", "timeout"):
            return n
    return None


def _largest_fitting(legs, mesh: str, path: str) -> int | None:
    fitting = [n for n in _sizes(legs) if (legs.get((mesh, path, n)) or Leg(mesh, path, n)).fits]
    return max(fitting) if fitting else None


def _exact_comparator(legs, mesh: str, n: int) -> tuple[float | None, str]:
    """Exact solve + two log-dets at this N, from the dense leg then the sparse."""
    for path in ("dense", "sparse"):
        leg = legs.get((mesh, path, n))
        if leg is not None and leg.status == "ok":
            total = leg.exact_total_ms()
            if total is not None:
                return total, path
    return None, ""


def render_crossover(legs: dict[tuple[str, str, int], Leg]) -> list[str]:
    lines = [
        "## 3. Crossover — computed from the rows above, never interpolated",
        "",
    ]
    for mesh in MESHES:
        if not any((mesh, path, n) in legs for path in PATHS for n in _sizes(legs)):
            continue
        lines.append(f"**{mesh}.**")
        sentences: list[str] = []

        for path, label in (("dense", "Dense"), ("sparse", "Sparse")):
            first_bad = _first_not_fitting(legs, mesh, path)
            largest = _largest_fitting(legs, mesh, path)
            if first_bad is not None:
                sentences.append(f"{label} stops fitting at N = {first_bad}.")
            elif largest is not None:
                sentences.append(
                    f"{label} fits at every measured N up to {largest}; where it stops is "
                    f"{NOT_MEASURED}."
                )
            else:
                sentences.append(f"{label}: {NOT_MEASURED}.")

        crossed_at: int | None = None
        comparisons: list[str] = []
        for n in _sizes(legs):
            mf = legs.get((mesh, "matrix_free", n))
            if mf is None or mf.status != "ok":
                continue
            mf_total = mf.per_call_ms()
            exact_total, exact_path = _exact_comparator(legs, mesh, n)
            if mf_total is None:
                continue
            if exact_total is None:
                comparisons.append(f"N = {n}: no exact comparator at this N")
                continue
            verb = "beats" if mf_total < exact_total else "loses to"
            comparisons.append(
                f"N = {n}: matrix-free {mf_total:.3f} ms {verb} the {exact_path} exact "
                f"{exact_total:.3f} ms"
            )
            if mf_total < exact_total and crossed_at is None:
                crossed_at = n
        if comparisons:
            sentences.append(
                f"Matrix-free (unconstrained PCG + SLQ m={SLQ_STEPS}) vs the exact solve + two "
                f"log-det Choleskys — {'; '.join(comparisons)}."
            )
            sentences.append(
                f"First measured N where matrix-free wins: {crossed_at}."
                if crossed_at is not None
                else "Matrix-free does not win at any measured N; a crossover above the largest "
                f"measured N is {NOT_MEASURED}."
            )
        else:
            sentences.append(f"Matrix-free per-call vs exact: {NOT_MEASURED}.")

        errors: list[str] = []
        for n in _sizes(legs):
            mf = legs.get((mesh, "matrix_free", n))
            if mf is None or mf.status != "ok":
                continue
            err = mf.log_evidence_error()
            if err is None:
                continue
            verdict = "within" if abs(err) <= 0.5 else "outside"
            errors.append(f"N = {n}: {err:+.3f} nats ({verdict} the 0.5-nat bar)")
        sentences.append(
            "SLQ log-evidence error — " + "; ".join(errors) + "."
            if errors
            else f"SLQ log-evidence error: {NOT_MEASURED}."
        )

        lines.extend(f"- {sentence}" for sentence in sentences)
        lines.append("")
    return lines


def render(
    results_dir: Path,
    manifest_path: Path,
    logs_dir: Path,
) -> str:
    legs, manifest = collect(results_dir, manifest_path, logs_dir)
    header = [
        "# Reduced N_src size sweep — autolens_profiling#247",
        "",
        f"- results: `{results_dir}`",
        f"- manifest: `{manifest_path}`"
        + ("" if manifest else " (absent — legs discovered from logs/JSONs)"),
        f"- logs: `{logs_dir}`",
        f"- legs known: {len(legs)}"
        + (f" ({len(manifest)} submitted per the manifest)" if manifest else ""),
        "",
    ]
    body = render_fits_table(legs) + render_per_call_table(legs) + render_crossover(legs)
    return "\n".join(header + body).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "breakdown" / "imaging",
        help="where the per-leg result JSONs live",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "sweep" / "size_sweep_manifest.jsonl",
        help="the launcher's submitted-leg manifest (optional)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=ROOT / "hpc" / "batch_gpu",
        help="directory holding output/ and error/ SLURM logs",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the markdown here")
    args = parser.parse_args(argv)

    markdown = render(args.results_dir, args.manifest, args.logs_dir)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
