"""W7 — CP-4 follow-up: per-draw attribution of the AdaptSplit NaN wall (autolens_profiling#164).

`slogdet_ab.py` (Phase 8A, CP-4) measured that on `delaunay_adapt_split` slogdet
rescues 64-73% of the cholesky NaNs with zero regressions, but 20-32 draws are
NaN under BOTH arms, all 128 lambda_transect draws have non-finite gradients
under both arms, and the marginal-band value delta is tier-dependent (up to
9,619 nats on A100 vs 2.27 on RAL CPU). That script's verdict is a population
count; it cannot say WHY any one draw failed. This script replays individual
stored draws non-jitted, drills into `inversion.*` and small `jax.grad`
closures, and assigns each draw one mechanism label so the population counts
in RESULTS.md can be read against a cause rather than a symptom.

Established (verify against the loaded artifacts, then extend):

1. 10 of the A100 tier's 32 nan-under-both draws (indices 406-415, all
   `descent`) have a NaN INPUT VECTOR (dead lanes captured by the descent
   checkpoint spy, `slogdet_ab.py` ~478-536); six more descent draws
   (384, 391, 393, 394, 397, 405) carry a NEGATIVE physical regularization
   coefficient (e.g. draw 405 outer_coefficient=-5.22). So
   `marginal_band.coefficient_min=-5.2` in `RESULTS.md`'s prior "Ops notes" is
   a harvest bug (an unfiltered descent lane), not a log axis.
2. All 128 `lambda_transect` draws sit at the anchor (prior medians), where
   `ell_comps` and shear are EXACTLY (0, 0). `autogalaxy/convert.py:86`
   (`axis_ratio_and_angle_from`) and `:220` (`shear_magnitude_and_angle_from`)
   both take `sqrt(e1**2 + e2**2)`, whose gradient is undefined at the origin
   — a driver artefact, not evidence about the regularization wall.
3. The remaining nan-under-both draws are `prior` draws with a regularization
   coefficient in ~4e5-9e5 (`adapt.py:47` builds a linear weight,
   `regularization_util.py:257`/`336` square it into the matrix — the "lambda
   to the 4th" fragility of #104).
4. Tier dependence is per-draw, not per-population: A100 draw 96 (prior,
   coefficients [3.5e5, 536]) is the 9,619-nat marginal-band outlier on A100
   and is NaN-under-both on the native RAL CPU tier for the IDENTICAL input
   vector.

Draw classes (see `_select_indices_for_class`):

    nan_both        NaN under both arms in the stored replay (per_draw.npz)
    rescued         NaN under cholesky, finite under slogdet
    marginal_band   the JSON verdict's `marginal_band.draw_indices`
    transect        ~16 lambda_transect draws subsampled evenly across the array
    tier_flip       finite on one tier / NaN on the other, over the SHARED
                    draw prefix (prior + truth_bar + lambda_transect were
                    harvested with the same --seed on both tiers, so the
                    first `min(n_a100, n_cpu)` rows are the identical vectors
                    — verified by `np.allclose` before use)
    truth_bar       ~8 truth_bar draws, control group (should mostly be clean)

Classification (one label per draw, first match wins; computed once per draw
from the CONTROL (cholesky) arm's inversion, since `curvature_reg_matrix_reduced`
does not depend on `log_det_method` — the two arms differ only in which scalar
formula reads that matrix):

    dead_lane               input vector is entirely non-finite
    invalid_coefficient     a regularization coefficient is below its own
                             prior's lower limit (a harvest bug, see (1) above)
    anchor_singularity      figure_of_merit finite, ell_comps-only jax.grad
                             non-finite, coefficient-only jax.grad finite, and
                             an exact-zero ell_comps/shear component is present
    upstream_nan            mapping_matrix / data_vector / curvature_matrix
                             non-finite BEFORE any log-det is computed
    genuinely_singular      inputs finite and (cond(F+H) >= 1e16, OR the numpy
                             LAPACK cholesky raises LinAlgError and np.slogdet's
                             sign != +1 or logabsdet is non-finite)
    marginal_tier_flippable 1e12 <= cond(F+H) < 1e16
    clean                   none of the above fired (cond < 1e12, finite)
    build_error:<message>   fit_from / inversion construction raised

Usage (from the ``autolens_profiling/`` root)::

    python3 scripts/misc/searches/slogdet_nan_attribution.py \\
        --tier-file a100 --classes nan_both,transect --max-per-class 4
    python3 scripts/misc/searches/slogdet_nan_attribution.py \\
        --tier-file ral_cpu --classes all
    python3 scripts/misc/searches/slogdet_nan_attribution.py \\
        --tier-file a100 --classes tier_flip,marginal_band --matrix-only

Writes:

    results/notes/inference/phase_08_regularization/slogdet_ab/attribution/<cell>_<tier>.json
    results/notes/inference/phase_08_regularization/slogdet_ab/attribution/<cell>_<tier>.npz
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
for _p in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

import numpy as np  # noqa: E402
from searches._setup import build_ab_for_cell  # noqa: E402

CONTROL = "cholesky"
TREATMENT = "slogdet"

# The regularization coefficients whose conditioning produces the wall (same
# names slogdet_ab.py sweeps on the lambda transect).
_REG_COEFFICIENT_NAMES = ("inner_coefficient", "outer_coefficient")
# Names whose sqrt-magnitude conversion (autogalaxy/convert.py) is undefined at
# exactly (0, 0) — the anchor-singularity mechanism.
_ANCHOR_SENSITIVE_SUFFIXES = ("ell_comps_0", "ell_comps_1", "gamma_1", "gamma_2")

_ALL_CLASSES = ("nan_both", "rescued", "marginal_band", "transect", "tier_flip", "truth_bar")

# Classification thresholds on cond(curvature_reg_matrix_reduced), read off
# float64's ~1e16 precision floor.
_COND_SINGULAR = 1e16
_COND_MARGINAL_LOW = 1e12
_REG_FLOOR_VALUE = 2e-8
_REG_FLOOR_ATOL = 1e-12


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset-class", default="imaging")
    p.add_argument("--model-type", default="delaunay_adapt_split")
    p.add_argument("--instrument", default="hst")
    p.add_argument(
        "--tier-file",
        required=True,
        choices=("a100", "ral_cpu"),
        help="which stored slogdet_ab draws-tag / tier artifact to replay",
    )
    p.add_argument(
        "--classes",
        default="all",
        help=f"comma-separated subset of {_ALL_CLASSES} or 'all'",
    )
    p.add_argument("--max-per-class", type=int, default=8)
    p.add_argument(
        "--matrix-only",
        action="store_true",
        help="skip the fit_from() gradient closures; only form "
        "curvature_reg_matrix_reduced and compare LAPACK cholesky / "
        "np.linalg.slogdet / eigvalsh-reference / jnp cholesky-slogdet "
        "on it. Cheaper — use for the tier_flip / marginal_band cond scan.",
    )
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def cell_name(args) -> str:
    return f"{args.dataset_class}_{args.model_type}_{args.instrument}"


def results_dir(args) -> _Path:
    if args.out_dir:
        return _Path(args.out_dir)
    return _ROOT / "results" / "notes" / "inference" / "phase_08_regularization" / "slogdet_ab"


def _tier_json_path(args, tier: str) -> _Path:
    matches = sorted(results_dir(args).glob(f"{cell_name(args)}_{tier}_*.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one verdict JSON for tier={tier!r} matching "
            f"{cell_name(args)}_{tier}_*.json under {results_dir(args)}, found "
            f"{len(matches)}: {matches}"
        )
    return matches[0]


def _tier_per_draw_path(args, tier: str) -> _Path:
    return _tier_json_path(args, tier).with_suffix(".per_draw.npz")


def _tier_draws_path(args, tier: str) -> _Path:
    return results_dir(args) / "draws" / f"{cell_name(args)}_{tier}.npz"


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


class TierData:
    def __init__(self, args, tier: str):
        self.tier = tier
        self.json_path = _tier_json_path(args, tier)
        self.verdict = json.loads(self.json_path.read_text())
        pd = np.load(_tier_per_draw_path(args, tier), allow_pickle=False)
        self.vectors = pd["vectors"]
        self.sources = pd["sources"].astype(str)
        self.cholesky_values = pd["cholesky_values"]
        self.slogdet_values = pd["slogdet_values"]
        self.cholesky_grad_finite = pd["cholesky_grad_finite"]
        self.slogdet_grad_finite = pd["slogdet_grad_finite"]
        draws = np.load(_tier_draws_path(args, tier), allow_pickle=True)
        self.parameter_names = draws["parameter_names"].astype(str).tolist()
        self.coefficient_indices = draws["coefficient_indices"].tolist()

    @property
    def n(self) -> int:
        return len(self.vectors)


# -----------------------------------------------------------------------------
# Draw-class selection
# -----------------------------------------------------------------------------


def _subsample(indices: np.ndarray, max_n: int) -> np.ndarray:
    indices = np.asarray(sorted(set(indices.tolist())))
    if max_n <= 0 or len(indices) <= max_n:
        return indices
    pick = np.unique(np.linspace(0, len(indices) - 1, max_n).round().astype(int))
    return indices[pick]


def _select_indices_for_class(cls: str, this: TierData, other: TierData | None, max_n: int):
    c_fin = np.isfinite(this.cholesky_values)
    t_fin = np.isfinite(this.slogdet_values)

    if cls == "nan_both":
        return _subsample(np.flatnonzero((~c_fin) & (~t_fin)), max_n)
    if cls == "rescued":
        return _subsample(np.flatnonzero((~c_fin) & t_fin), max_n)
    if cls == "marginal_band":
        idx = np.asarray(this.verdict["verdict"]["marginal_band"]["draw_indices"], dtype=int)
        return _subsample(idx, max_n)
    if cls == "transect":
        idx = np.flatnonzero(this.sources == "lambda_transect")
        return _subsample(idx, max_n)
    if cls == "truth_bar":
        idx = np.flatnonzero((this.sources == "truth_bar") & c_fin & t_fin)
        return _subsample(idx, max_n)
    if cls == "tier_flip":
        if other is None:
            print("  [tier_flip] other tier not available — skipping this class")
            return np.asarray([], dtype=int)
        n_common = min(this.n, other.n)
        if not np.allclose(this.vectors[:n_common], other.vectors[:n_common]):
            print(
                "  [tier_flip] the two tiers' first "
                f"{n_common} draws are not bit-identical — skipping (harvest seeds diverged)"
            )
            return np.asarray([], dtype=int)
        oc_fin = np.isfinite(other.cholesky_values[:n_common])
        ot_fin = np.isfinite(other.slogdet_values[:n_common])
        control_flip = c_fin[:n_common] != oc_fin
        treatment_flip = t_fin[:n_common] != ot_fin
        return _subsample(np.flatnonzero(control_flip | treatment_flip), max_n)
    raise ValueError(f"unknown class {cls!r}")


# -----------------------------------------------------------------------------
# Per-draw diagnostics
# -----------------------------------------------------------------------------


def _anchor_sensitive_indices(parameter_names: list[str]) -> list[int]:
    return [
        i
        for i, name in enumerate(parameter_names)
        if name.split(".")[-1] in _ANCHOR_SENSITIVE_SUFFIXES
    ]


def _matrix_diagnostics(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    finite = bool(np.isfinite(matrix).all())
    out = {
        "finite": finite,
        "min": float(np.nanmin(matrix)) if matrix.size else None,
        "max": float(np.nanmax(matrix)) if matrix.size else None,
    }
    if not finite:
        out.update(
            eig_min=None,
            eig_max=None,
            cond=None,
            lapack_cholesky_ok=None,
            lapack_cholesky_error=None,
            np_slogdet_sign=None,
            np_slogdet_logabsdet=None,
            eigvalsh_logdet=None,
        )
        return out

    try:
        eigvals = np.linalg.eigvalsh(matrix)
        eig_min, eig_max = float(eigvals.min()), float(eigvals.max())
        cond = eig_max / eig_min if eig_min > 0 else float("inf")
    except np.linalg.LinAlgError as exc:
        eigvals, eig_min, eig_max, cond = None, None, None, None
        out["eigvalsh_error"] = str(exc)

    out["eig_min"] = eig_min
    out["eig_max"] = eig_max
    out["cond"] = cond
    out["eigvalsh_logdet"] = (
        float(np.sum(np.log(eigvals))) if eigvals is not None and np.all(eigvals > 0) else None
    )

    try:
        chol = np.linalg.cholesky(matrix)
        out["lapack_cholesky_ok"] = True
        out["lapack_cholesky_error"] = None
        out["lapack_cholesky_logdet"] = float(2.0 * np.sum(np.log(np.diag(chol))))
    except np.linalg.LinAlgError as exc:
        out["lapack_cholesky_ok"] = False
        out["lapack_cholesky_error"] = str(exc)
        out["lapack_cholesky_logdet"] = None

    sign, logabsdet = np.linalg.slogdet(matrix)
    out["np_slogdet_sign"] = float(sign)
    out["np_slogdet_logabsdet"] = float(logabsdet) if np.isfinite(logabsdet) else None
    return out


def _make_grad_fn(model, analysis, indices: list[int]):
    """Build (and JIT-compile) ONE ``value_and_grad`` closure for a fixed
    parameter subset, to be reused across every draw.

    The whole draw-level analysis (``_arm_summary``) deliberately calls
    ``fit_from`` eagerly/non-jitted so the individual inversion matrices can be
    pulled out and inspected — fine for a handful of calls. The gradient probe
    is different: it is called twice per selected draw (anchor subset,
    coefficient subset), and eager per-op dispatch through a ~1500-source-pixel
    curvature/regularization solve and its VJP costs minutes, not seconds
    (RESULTS.md already records a 45->100s CPU *batched* compile for the much
    simpler forward-only evaluator in ``slogdet_ab.py``). The fix is to make
    the compile a FIXED cost rather than a per-draw one: ``full_vector`` is a
    traced argument, not a Python-closure constant, so the SAME compiled
    program is reused for every draw that shares this (model, arm, indices)
    triple — one compile total, not one per draw.

    Returns a callable ``grad_fn(vector) -> dict`` (``None`` if ``indices`` is
    empty).
    """
    if not indices:
        return None
    import jax
    import jax.numpy as jnp

    idx_arr = jnp.asarray(indices)

    def f(full_vector, sub):
        full = full_vector.at[idx_arr].set(sub)
        instance = model.instance_from_vector(vector=full, xp=jnp)
        fit = analysis.fit_from(instance)
        return fit.figure_of_merit

    compiled = jax.jit(jax.value_and_grad(f, argnums=1))

    def grad_fn(vector: np.ndarray) -> dict:
        full = jnp.asarray(vector)
        sub0 = full[idx_arr]
        try:
            value, grad = compiled(full, sub0)
            value = float(value)
            grad = np.asarray(grad)
            return {
                "value": value if np.isfinite(value) else None,
                "value_finite": bool(np.isfinite(value)),
                "grad": grad.tolist(),
                "grad_finite": bool(np.all(np.isfinite(grad))),
            }
        except Exception as exc:  # noqa: BLE001 — this is a diagnostic probe
            return {"value": None, "grad_finite": False, "error": f"{type(exc).__name__}: {exc}"}

    return grad_fn


def _arm_summary(analysis, model, vector: np.ndarray) -> dict:
    """Full per-arm inversion/fit diagnostics for one draw, one arm."""
    instance = model.instance_from_vector(vector=vector, xp=np)
    fit = analysis.fit_from(instance)
    inv = fit.inversion

    mapping_matrix = np.asarray(inv.mapping_matrix)
    data_vector = np.asarray(inv.data_vector)
    curvature_matrix = np.asarray(inv.curvature_matrix)
    reconstruction = np.asarray(inv.reconstruction)

    def _fm(v):
        v = float(v)
        return v if np.isfinite(v) else None

    return {
        "mapping_matrix_finite": bool(np.isfinite(mapping_matrix).all()),
        "mapping_matrix_min": float(np.nanmin(mapping_matrix)) if mapping_matrix.size else None,
        "mapping_matrix_max": float(np.nanmax(mapping_matrix)) if mapping_matrix.size else None,
        "data_vector_finite": bool(np.isfinite(data_vector).all()),
        "curvature_matrix_finite": bool(np.isfinite(curvature_matrix).all()),
        "reconstruction_finite": bool(np.isfinite(reconstruction).all()),
        "chi_squared": _fm(fit.chi_squared),
        "regularization_term": _fm(inv.regularization_term),
        "log_det_curvature_reg_matrix_term": _fm(inv.log_det_curvature_reg_matrix_term),
        "log_det_regularization_matrix_term": _fm(inv.log_det_regularization_matrix_term),
        "figure_of_merit": _fm(fit.figure_of_merit),
    }, inv


def _classify(
    *,
    vector_finite: bool,
    invalid_coefficient: bool,
    control_fom_finite: bool,
    anchor_grad: dict,
    coefficient_grad: dict,
    has_exact_zero_anchor_component: bool,
    upstream_nan: bool,
    matrix_diag: dict,
) -> str:
    if not vector_finite:
        return "dead_lane"
    if invalid_coefficient:
        return "invalid_coefficient"
    if (
        control_fom_finite
        and anchor_grad.get("grad_finite") is False
        and coefficient_grad.get("grad_finite") is True
        and has_exact_zero_anchor_component
    ):
        return "anchor_singularity"
    if upstream_nan:
        return "upstream_nan"
    cond = matrix_diag.get("cond")
    lapack_ok = matrix_diag.get("lapack_cholesky_ok")
    slogdet_sign = matrix_diag.get("np_slogdet_sign")
    slogdet_logabsdet = matrix_diag.get("np_slogdet_logabsdet")
    slogdet_bad = (slogdet_sign is not None and slogdet_sign != 1.0) or slogdet_logabsdet is None
    if cond is not None and cond >= _COND_SINGULAR:
        return "genuinely_singular"
    if lapack_ok is False and slogdet_bad:
        return "genuinely_singular"
    if cond is not None and _COND_MARGINAL_LOW <= cond < _COND_SINGULAR:
        return "marginal_tier_flippable"
    return "clean"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)

    import jax

    x64 = jax.numpy.zeros(1, dtype=jax.numpy.float64).dtype == np.float64
    if not x64:
        raise SystemExit("JAX_ENABLE_X64 is not set — run with JAX_ENABLE_X64=True.")

    classes = _ALL_CLASSES if args.classes == "all" else tuple(args.classes.split(","))
    unknown = set(classes) - set(_ALL_CLASSES)
    if unknown:
        raise SystemExit(f"unknown class(es) {unknown}; choose from {_ALL_CLASSES}")

    this = TierData(args, args.tier_file)
    other_tier = "ral_cpu" if args.tier_file == "a100" else "a100"
    try:
        other = TierData(args, other_tier)
    except SystemExit:
        other = None
        print(f"  (other tier {other_tier!r} artifact not found — tier_flip class will be empty)")

    print(f"=== ATTRIBUTION — {cell_name(args)} tier={args.tier_file} ===", flush=True)
    selected: dict[int, set[str]] = {}
    for cls in classes:
        idx = _select_indices_for_class(cls, this, other, args.max_per_class)
        print(f"  class {cls:16s} -> {len(idx)} draw(s): {idx.tolist()}")
        for i in idx.tolist():
            selected.setdefault(i, set()).add(cls)

    dataset, model, analyses = build_ab_for_cell(
        dataset_class=args.dataset_class,
        model_type=args.model_type,
        instrument=args.instrument,
        use_jax=True,
        use_mixed_precision=False,
        log_det_methods=(CONTROL, TREATMENT),
    )

    priors = list(model.priors_ordered_by_id)
    coefficient_lowers = [float(priors[i].lower_limit) for i in this.coefficient_indices]
    anchor_indices = _anchor_sensitive_indices(this.parameter_names)

    grad_fns = {}
    if not args.matrix_only and selected:
        print(
            "  compiling gradient probes (fixed one-time cost, reused across every draw)...",
            flush=True,
        )
        for arm in (CONTROL, TREATMENT):
            grad_fns[("anchor", arm)] = _make_grad_fn(model, analyses[arm], anchor_indices)
            grad_fns[("coefficient", arm)] = _make_grad_fn(
                model, analyses[arm], this.coefficient_indices
            )
        print("  ...compiled.", flush=True)

    records = []
    for n_done, draw_idx in enumerate(sorted(selected)):
        classes_hit = sorted(selected[draw_idx])
        vector = np.asarray(this.vectors[draw_idx], dtype=float)
        vector_finite = bool(np.isfinite(vector).all())
        print(
            f"  [{n_done + 1}/{len(selected)}] draw {draw_idx} "
            f"(source={this.sources[draw_idx]}, classes={classes_hit})",
            flush=True,
        )

        coefficients = (
            vector[this.coefficient_indices].tolist()
            if vector_finite
            else [None] * len(this.coefficient_indices)
        )
        invalid_coefficient = vector_finite and any(
            c < lo for c, lo in zip(coefficients, coefficient_lowers)
        )
        anchor_values = vector[anchor_indices].tolist() if vector_finite else []
        has_exact_zero_anchor_component = vector_finite and any(v == 0.0 for v in anchor_values)

        record: dict = {
            "draw_index": int(draw_idx),
            "source": str(this.sources[draw_idx]),
            "classes": classes_hit,
            "vector_finite": vector_finite,
            "coefficients": coefficients,
            "coefficient_lowers": coefficient_lowers,
            "invalid_coefficient": invalid_coefficient,
            "anchor_component_values": anchor_values,
            "has_exact_zero_anchor_component": has_exact_zero_anchor_component,
        }

        if not vector_finite:
            record["classification"] = "dead_lane"
            records.append(record)
            continue

        try:
            control_summary, control_inv = _arm_summary(analyses[CONTROL], model, vector)
        except Exception as exc:  # noqa: BLE001
            record["classification"] = f"build_error:{type(exc).__name__}: {exc}"
            records.append(record)
            continue

        matrix = np.asarray(control_inv.curvature_reg_matrix_reduced)
        matrix_diag = _matrix_diagnostics(matrix)
        reg_matrix = np.asarray(control_inv.regularization_matrix_reduced)
        reg_diag = np.diag(reg_matrix) if reg_matrix.ndim == 2 else np.asarray([])
        floor_fraction = (
            float(np.mean(np.isclose(reg_diag, _REG_FLOOR_VALUE, atol=_REG_FLOOR_ATOL)))
            if reg_diag.size
            else None
        )

        upstream_nan = not (
            control_summary["mapping_matrix_finite"]
            and control_summary["data_vector_finite"]
            and control_summary["curvature_matrix_finite"]
        )

        record["control"] = control_summary
        record["regularization_matrix_reduced_floor_fraction"] = floor_fraction
        record["curvature_reg_matrix_reduced"] = matrix_diag
        record["upstream_nan"] = upstream_nan

        if invalid_coefficient:
            record["classification"] = "invalid_coefficient"
            records.append(record)
            continue

        if args.matrix_only:
            record["treatment_log_det_curvature_reg_matrix_term"] = None
            try:
                treatment_instance = model.instance_from_vector(vector=vector, xp=np)
                treatment_fit = analyses[TREATMENT].fit_from(treatment_instance)
                v = float(treatment_fit.inversion.log_det_curvature_reg_matrix_term)
                record["treatment_log_det_curvature_reg_matrix_term"] = (
                    v if np.isfinite(v) else None
                )
            except Exception as exc:  # noqa: BLE001
                record["treatment_error"] = f"{type(exc).__name__}: {exc}"
            record["classification"] = _classify(
                vector_finite=True,
                invalid_coefficient=False,
                control_fom_finite=control_summary["figure_of_merit"] is not None,
                anchor_grad={},
                coefficient_grad={},
                has_exact_zero_anchor_component=has_exact_zero_anchor_component,
                upstream_nan=upstream_nan,
                matrix_diag=matrix_diag,
            )
            records.append(record)
            continue

        try:
            treatment_summary, _ = _arm_summary(analyses[TREATMENT], model, vector)
        except Exception as exc:  # noqa: BLE001
            treatment_summary = {"error": f"{type(exc).__name__}: {exc}"}
        record["treatment"] = treatment_summary

        anchor_grad = {
            arm: (
                grad_fns[("anchor", arm)](vector)
                if grad_fns[("anchor", arm)]
                else {"note": "no indices"}
            )
            for arm in (CONTROL, TREATMENT)
        }
        coefficient_grad = {
            arm: grad_fns[("coefficient", arm)](vector) for arm in (CONTROL, TREATMENT)
        }
        record["anchor_grad"] = anchor_grad
        record["coefficient_grad"] = coefficient_grad

        record["classification"] = _classify(
            vector_finite=True,
            invalid_coefficient=False,
            control_fom_finite=control_summary["figure_of_merit"] is not None,
            anchor_grad=anchor_grad[CONTROL],
            coefficient_grad=coefficient_grad[CONTROL],
            has_exact_zero_anchor_component=has_exact_zero_anchor_component,
            upstream_nan=upstream_nan,
            matrix_diag=matrix_diag,
        )
        records.append(record)

    summary_by_class: dict[str, dict[str, int]] = {}
    for rec in records:
        cls_label = rec["classification"]
        for cls in rec["classes"]:
            summary_by_class.setdefault(cls, {}).setdefault(cls_label, 0)
            summary_by_class[cls][cls_label] += 1

    classification_totals: dict[str, int] = {}
    for rec in records:
        classification_totals[rec["classification"]] = (
            classification_totals.get(rec["classification"], 0) + 1
        )

    out = {
        "schema_version": 1,
        "experiment": "w7_slogdet_nan_attribution",
        "issue": "autolens_profiling#164",
        "cell": cell_name(args),
        "tier_file": args.tier_file,
        "other_tier_available": other is not None,
        "matrix_only": args.matrix_only,
        "classes_requested": list(classes),
        "max_per_class": args.max_per_class,
        "n_draws_analyzed": len(records),
        "classification_totals": classification_totals,
        "summary_by_class": summary_by_class,
        "anchor_sensitive_parameter_indices": anchor_indices,
        "anchor_sensitive_parameter_names": [this.parameter_names[i] for i in anchor_indices],
        "coefficient_indices": this.coefficient_indices,
        "coefficient_parameter_names": [this.parameter_names[i] for i in this.coefficient_indices],
        "thresholds": {
            "cond_singular": _COND_SINGULAR,
            "cond_marginal_low": _COND_MARGINAL_LOW,
            "regularization_floor_value": _REG_FLOOR_VALUE,
        },
        "draws": records,
    }

    out_dir = results_dir(args) / "attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_matrix_only" if args.matrix_only else ""
    dest = out_dir / f"{cell_name(args)}_{args.tier_file}{suffix}.json"
    dest.write_text(json.dumps(out, indent=2, default=str) + "\n")

    npz_dest = dest.with_suffix(".npz")
    np.savez(
        npz_dest,
        draw_index=np.asarray([r["draw_index"] for r in records], dtype=int),
        classification=np.asarray([r["classification"] for r in records]),
        cond=np.asarray(
            [
                (r.get("curvature_reg_matrix_reduced", {}) or {}).get("cond")
                if r.get("curvature_reg_matrix_reduced", {}).get("cond") is not None
                else np.nan
                for r in records
            ],
            dtype=float,
        ),
    )

    print("\n" + "=" * 78)
    print(f"CLASSIFICATION TOTALS ({args.tier_file}, matrix_only={args.matrix_only})")
    print("=" * 78)
    for label, n in sorted(classification_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {label}")
    print(f"\nwritten: {dest}")
    print(f"written: {npz_dest}")


if __name__ == "__main__":
    main()
