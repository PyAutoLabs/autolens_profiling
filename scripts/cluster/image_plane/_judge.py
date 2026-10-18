"""
Cluster PointSolver verdicts: one source's solver candidates judged against the oracle.

Split out of ``robust_settings_sweep.py`` (autolens_profiling#436) so the verdict
logic and the per-draw JSON entry can be imported without running the sweep, by
``oracle_figure.py`` and the tests. NumPy only; no PyAutoLens import.

``judge`` splits one source's accepted oracle roots by magnification against the
solver threshold, matches the distinct candidates to them within ``tol`` and
returns the counts plus the full ``coverage`` / ``spurious`` / ``ignored`` lists.
``per_draw_source_entry`` reduces that to the bounded JSON entry a row keeps per
draw and source: the count keys (unchanged; downstream tables read them) plus the
positions needed to draw the verdict (see ``POSITION_KEYS``).
"""

import numpy as np

MAGNIFICATION_THRESHOLD = 0.1
MU_AMBIGUITY_BAND = 0.10
DISTINCT_TOL = 0.005

# Per-draw cap on recorded spurious / ignored candidate positions (counts stay exact).
MAX_RECORDED_CANDIDATES = 20

COUNT_KEYS = (
    "n_roots_considered",
    "n_candidates_raw",
    "n_candidates_distinct",
    "n_covered",
    "n_missed",
    "n_duplicated",
    "n_spurious",
    "multiplicity_match",
    "max_position_error",
    "max_candidate_residual",
)
POSITION_KEYS = ("tol", "roots", "spurious", "duplicated_candidates", "ignored_candidates")
VERDICTS = ("covered", "missed", "duplicated")


def finite_positions(p):
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    return p[np.all(np.isfinite(p), axis=1)]


def distinct(p, tol=DISTINCT_TOL):
    out = []
    for q in p:
        if not any(np.hypot(*(q - r)) < tol for r in out):
            out.append(q)
    return np.asarray(out).reshape(-1, 2)


def _in_box(p, box):
    return box[0] <= p[0] <= box[1] and box[2] <= p[1] <= box[3]


def judge(oracle_source, candidates, tol, box, eq, plane_index, beta) -> dict:
    """Per-root coverage, multiplicity and per-candidate residual/convergence for one source.

    ``eq`` needs only ``residual(positions (N, 2), plane_index, beta) -> (N, 2)``.
    """
    roots = [r for r in oracle_source["roots"] if r["accepted"]]
    lo = MAGNIFICATION_THRESHOLD / (1 + MU_AMBIGUITY_BAND)
    hi = MAGNIFICATION_THRESHOLD * (1 + MU_AMBIGUITY_BAND)
    considered, ambiguous, below = [], [], []
    for r in roots:
        m = abs(r["magnification"])
        (considered if m > hi else ambiguous if m >= lo else below).append(r)
    cands = distinct(finite_positions(candidates))
    n_raw = int(finite_positions(candidates).shape[0])
    rpos = np.asarray([r["position"] for r in considered]).reshape(-1, 2)
    other = np.asarray([r["position"] for r in ambiguous + below]).reshape(-1, 2)
    coverage, errors = [], []
    for r in considered:
        p = np.asarray(r["position"])
        d = np.hypot(*(cands - p).T) if cands.size else np.zeros(0)
        within = np.flatnonzero(d <= tol)
        n_match = int(within.size)
        coverage.append(
            {
                "position": r["position"],
                "magnification": r["magnification"],
                "parity": r.get("parity"),
                "sigma_min": r["singular_values"][1],
                "inside_extent": _in_box(p, box),
                "n_candidates_within_tol": n_match,
                "verdict": "covered"
                if n_match == 1
                else ("missed" if n_match == 0 else "duplicated"),
                "nearest_candidate_distance": float(d.min()) if d.size else None,
                "matched": [
                    {"position": cands[i].tolist(), "error": float(d[i])}
                    for i in within[np.argsort(d[within])]
                ],
            }
        )
        if n_match >= 1:
            errors.append(float(d.min()))
    spurious, ignored, residuals = [], [], []
    if cands.size:
        res = np.hypot(*eq.residual(cands, plane_index, beta).T)
        for k, c in enumerate(cands):
            d_main = np.hypot(*(rpos - c).T).min() if rpos.size else np.inf
            d_other = np.hypot(*(other - c).T).min() if other.size else np.inf
            residuals.append(float(res[k]))
            if d_main <= tol:
                continue
            if d_other <= tol:
                # a candidate for a below-threshold / ambiguous root
                ignored.append({"position": c.tolist(), "nearest_root_distance": float(d_other)})
                continue
            spurious.append(
                {
                    "position": c.tolist(),
                    "residual": float(res[k]),
                    "nearest_root_distance": float(min(d_main, d_other)),
                }
            )
    n_relevant = int(cands.shape[0]) - len(ignored)
    missed = sum(c["verdict"] == "missed" for c in coverage)
    duplicated = sum(c["verdict"] == "duplicated" for c in coverage)
    return {
        "tol": float(tol),
        "n_roots_considered": len(considered),
        "n_roots_threshold_ambiguous": len(ambiguous),
        "n_roots_below_threshold": len(below),
        "n_candidates_raw": n_raw,
        "n_candidates_distinct": int(cands.shape[0]),
        "n_candidates_for_ignored_roots": len(ignored),
        "n_covered": len(considered) - missed - duplicated,
        "n_missed": int(missed),
        "n_missed_outside_extent": int(
            sum(c["verdict"] == "missed" and not c["inside_extent"] for c in coverage)
        ),
        "n_duplicated": int(duplicated),
        "n_spurious": len(spurious),
        "multiplicity_match": n_relevant == len(considered),
        "unresolved": int(missed + len(spurious)),
        "max_position_error": max(errors) if errors else None,
        "max_candidate_residual": max(residuals) if residuals else None,
        "coverage": coverage,
        "spurious": spurious[:5],
        "spurious_all": spurious,
        "ignored": ignored,
    }


def per_draw_source_entry(j: dict) -> dict:
    """The bounded JSON entry a row keeps per draw and source (``None`` passes through).

    Count keys (``COUNT_KEYS``) are unchanged. Position keys (``POSITION_KEYS``):

    - ``tol``: the row's match tolerance (arcsec);
    - ``roots``: every considered oracle root -- ``position``, ``magnification``,
      ``parity``, ``verdict`` (covered / missed / duplicated) and ``matched`` (each
      distinct candidate within ``tol``: ``position``, ``error``, nearest first);
    - ``spurious``: candidates within ``tol`` of no oracle root -- ``position``,
      ``residual``, ``nearest_root_distance`` (first ``MAX_RECORDED_CANDIDATES``);
    - ``duplicated_candidates``: the flat list of candidate positions matched to a
      duplicated root;
    - ``ignored_candidates``: candidates attributed to below-threshold / ambiguous
      roots (not judged) -- ``position``, ``nearest_root_distance``.
    """
    if j is None:
        return None
    entry = {k: j[k] for k in COUNT_KEYS}
    entry["tol"] = j["tol"]
    entry["roots"] = [
        {
            "position": c["position"],
            "magnification": c["magnification"],
            "parity": c.get("parity"),
            "verdict": c["verdict"],
            "matched": c["matched"],
        }
        for c in j["coverage"]
    ]
    entry["spurious"] = j["spurious_all"][:MAX_RECORDED_CANDIDATES]
    entry["duplicated_candidates"] = [
        m["position"] for c in j["coverage"] if c["verdict"] == "duplicated" for m in c["matched"]
    ]
    entry["ignored_candidates"] = j["ignored"][:MAX_RECORDED_CANDIDATES]
    return entry
