"""Truth-recovery scoring for the group4 MGE search benchmark.

The whole point of the ``searches/<sampler>/group/mge.py`` benchmark is not just
"how fast did the search run" but "**did it recover the input truth**". The
simulator (``simulators/group4_mge.py``) writes a ``truth.json`` next to the
dataset; this module compares a fit's ``max_log_likelihood_instance`` against it
and returns a JSON-friendly report that ``_runner`` embeds in the summary.

Recoverable quantities (MGE amplitudes are linear nuisance params, so light
intensities are *not* scored):

- per deflector: mass ``einstein_radius`` (fractional error) + mass ``centre``
  (Euclidean distance error, arcsec);
- primary deflector: external ``shear`` (``gamma_1``/``gamma_2`` abs error);
- per source: MGE basis ``centre`` (distance error, arcsec).

``overall_pass`` is ``True`` when every Einstein radius is within
``ER_RTOL`` and every mass/source centre within ``CENTRE_ATOL`` — a deliberately
lenient "found the right basin" bar, not a precision-cosmology one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# "Recovered the basin" tolerances — lenient on purpose (this scores whether an
# optimizer found the right lens model at all, not sub-percent accuracy).
ER_RTOL = 0.10  # Einstein radius: within 10%
CENTRE_ATOL = 0.10  # centres: within 0.1 arcsec
SHEAR_ATOL = 0.03  # shear components: within 0.03


def load_truth(dataset_path: Path) -> dict | None:
    """Read ``truth.json`` from the dataset dir; ``None`` if absent."""
    truth_file = Path(dataset_path) / "truth.json"
    if not truth_file.exists():
        return None
    return json.loads(truth_file.read_text())


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def recovery_report(instance: Any, truth: dict) -> dict:
    """Compare a max-log-likelihood ``instance`` against the ``truth`` record.

    Returns ``{"lenses": [...], "sources": [...], "overall_pass": bool,
    "max_einstein_radius_frac_error": float, "max_centre_error_arcsec": float}``.
    Every field is best-effort: a missing attribute records ``null`` for that
    entry rather than raising, so a partially-converged fit still scores.
    """
    galaxies = instance.galaxies
    lens_reports: list[dict] = []
    max_er_frac = 0.0
    max_centre_err = 0.0
    all_pass = True

    for lens in truth["lenses"]:
        name = lens["name"]
        entry: dict = {"name": name}
        gal = getattr(galaxies, name, None)
        truth_er = float(lens["einstein_radius"])
        truth_centre = tuple(lens["centre"])
        try:
            fit_er = float(gal.mass.einstein_radius)
            er_frac = abs(fit_er - truth_er) / max(truth_er, 1e-8)
            entry["einstein_radius_truth"] = truth_er
            entry["einstein_radius_fit"] = fit_er
            entry["einstein_radius_frac_error"] = er_frac
            max_er_frac = max(max_er_frac, er_frac)
            all_pass = all_pass and er_frac <= ER_RTOL
        except AttributeError:
            entry["einstein_radius_fit"] = None
            all_pass = False
        try:
            fit_centre = tuple(float(c) for c in gal.mass.centre)
            centre_err = _distance(fit_centre, truth_centre)
            entry["centre_truth"] = list(truth_centre)
            entry["centre_fit"] = list(fit_centre)
            entry["centre_error_arcsec"] = centre_err
            max_centre_err = max(max_centre_err, centre_err)
            all_pass = all_pass and centre_err <= CENTRE_ATOL
        except AttributeError:
            entry["centre_fit"] = None
            all_pass = False

        if lens.get("shear") is not None:
            try:
                shear = gal.shear
                g1_err = abs(float(shear.gamma_1) - float(lens["shear"][0]))
                g2_err = abs(float(shear.gamma_2) - float(lens["shear"][1]))
                entry["shear_truth"] = list(lens["shear"])
                entry["shear_fit"] = [float(shear.gamma_1), float(shear.gamma_2)]
                entry["shear_abs_error"] = [g1_err, g2_err]
                all_pass = all_pass and max(g1_err, g2_err) <= SHEAR_ATOL
            except AttributeError:
                entry["shear_fit"] = None
                all_pass = False
        lens_reports.append(entry)

    source_reports: list[dict] = []
    for source in truth["sources"]:
        name = source["name"]
        entry = {"name": name}
        gal = getattr(galaxies, name, None)
        truth_centre = tuple(source["centre"])
        try:
            fit_centre = tuple(float(c) for c in gal.bulge.centre)
            centre_err = _distance(fit_centre, truth_centre)
            entry["centre_truth"] = list(truth_centre)
            entry["centre_fit"] = list(fit_centre)
            entry["centre_error_arcsec"] = centre_err
            max_centre_err = max(max_centre_err, centre_err)
            all_pass = all_pass and centre_err <= CENTRE_ATOL
        except AttributeError:
            entry["centre_fit"] = None
            all_pass = False
        source_reports.append(entry)

    return {
        "lenses": lens_reports,
        "sources": source_reports,
        "overall_pass": bool(all_pass),
        "max_einstein_radius_frac_error": max_er_frac,
        "max_centre_error_arcsec": max_centre_err,
        "tolerances": {
            "einstein_radius_rtol": ER_RTOL,
            "centre_atol_arcsec": CENTRE_ATOL,
            "shear_atol": SHEAR_ATOL,
        },
    }
