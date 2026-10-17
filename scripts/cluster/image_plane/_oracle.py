"""
Independent lens-equation oracle for the cluster PointSolver robust-settings sweep
==================================================================================

The ground truth every verdict of ``robust_settings_sweep.py`` rests on
(autolens_profiling#436, Pulse campaign ``cluster-pointsolver`` phase 1). It is
independent of the triangle ``PointSolver``: no triangle tiling, no containment
test, no candidate grouping, no simulator output. It needs only the ``Tracer``'s
multi-plane ray-tracing (``Tracer.traced_grid_2d_list_from``, NumPy fp64) and
solves the lens equation for each source plane directly::

    r(theta) = traced_plane_j(theta) - beta = 0

Method
------

1. **Dense sampling.** ``r`` is sampled on a uniform lattice of spacing ``h``
   over the oracle box (larger than any solver extent under test, so a root a
   solver cannot reach is still known).
2. **Seeds, one per basin.** Every lattice cell whose four corners bracket zero
   in *both* components of ``r`` (a sign-change cell), and every local minimum
   of ``|r|`` over its 3 x 3 neighbourhood. No seed is ever discarded for being
   "far" from a root: Newton decides.
3. **Newton polish.** Damped Newton on ``r`` with a central finite-difference
   Jacobian (step ``FD_STEP``), backtracking line search, step length capped at
   ``4 h`` so a seed stays near its basin; iterated until ``|r|`` stops
   decreasing (round-off), never stopped at a residual target. A point is a
   *converged* root only when ``|r| < ACCEPT_RESIDUAL`` AND its final Newton
   correction ``|A^-1 r| < CERTIFY_STEP`` = 5e-8": near a cusp ``A`` is nearly
   singular (sigma_min ~ 7e-9 on the 1e-7 fixture), so a small residual alone
   does not localise a root -- the correction is the position uncertainty.
4. **Local refinement** (cusps and folds). Around every root whose Jacobian
   ``A = d traced / d theta`` has a small singular value (``< REFINE_SIGMA`` at
   depth 0, shrinking by ``REFINE_FACTOR`` per depth as the lattice does),
   whose nearest neighbour root is within ``8 h``, and around every local-minimum
   seed that failed to converge while ``|r| < h``, a finer lattice
   (``h / REFINE_FACTOR``, half-width ``6 h``) is sampled and steps 2-3 are
   repeated, recursively to ``REFINE_DEPTH`` levels. Merging images (a cusp
   triplet 7e-4" apart, the 1c/1d fixture) are resolved this way.
5. **Acceptance.** A root is accepted only when (a) its residual is at
   round-off, ``|r| < ACCEPT_RESIDUAL`` = 1e-10"; (b) the WHOLE procedure re-run
   at ``h / 2`` finds it again within ``STABILITY_TOL`` = 1e-6" (and finds no
   root the ``h`` run missed -- otherwise the source is flagged unstable); and
   (c) its magnification ``1 / det A`` is finite. Each root records its
   residual, signed magnification, parity, and the singular values of ``A``.

Calibration (``--calibrate``)
-----------------------------

Before any cluster row is judged the oracle must reproduce, to < 1e-6":

- the closed-form isothermal + external-shear quad of phase 1a
  (``autolens_workspace_test/scripts/point_source/solver/error_audit.py``,
  ``analytic_quad``: theta_E 1.6, shear 0.1, beta = 0; images at
  ``(+-b/(1+g), 0)``, ``(0, +-b/(1-g))``);
- the 1c/1d ``angular_roots`` fixture
  (``.../duplicate_policy.py``): the analytic-quad offset sources and the three
  near-caustic cusp sources (fractions 1e-3, 1e-5, 1e-7 of the cusp; the last
  has a 7.1e-4" minimum image separation). ``angular_roots`` is vendored below
  (it is a fixture-specific bracketing oracle valid only for isothermal +
  linear shear, not a general solver) so the calibration runs on a host without
  the workspace_test checkout; when that checkout is found, the upstream
  function is also run and both its output and the committed evidence JSON
  (``duplicate_policy_evidence.json``) are compared.

``python scripts/cluster/image_plane/_oracle.py --calibrate [--output PATH]``
prints the table and exits non-zero unless every case passes. The sweep embeds
:func:`calibrate`'s result in every JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ORACLE_VERSION = 1
ACCEPT_RESIDUAL = 1.0e-10
STABILITY_TOL = 1.0e-6
CALIBRATION_TOL = 1.0e-6
FD_STEP = 1.0e-5
NEWTON_MAX_ITER = 60
LINE_SEARCH_STEPS = 24
CERTIFY_STEP = 5.0e-8
NEWTON_FLOOR_STEP = 1.0e-13
DEDUP_TOL = 5.0e-7
REFINE_SIGMA = 0.3
REFINE_NEIGHBOUR_H = 8.0
REFINE_FACTOR = 16
REFINE_HALF_WIDTH_H = 6.0
REFINE_DEPTH = 3
REFINE_MAX_BOXES = 400
MAX_LOCAL_MINIMA = 20000
CHUNK = 400_000

PARAMETERS = {
    "oracle_version": ORACLE_VERSION,
    "accept_residual_arcsec": ACCEPT_RESIDUAL,
    "stability_tol_arcsec": STABILITY_TOL,
    "calibration_tol_arcsec": CALIBRATION_TOL,
    "fd_step_arcsec": FD_STEP,
    "newton_max_iter": NEWTON_MAX_ITER,
    "line_search_steps": LINE_SEARCH_STEPS,
    "certify_newton_correction_arcsec": CERTIFY_STEP,
    "newton_floor_step_arcsec": NEWTON_FLOOR_STEP,
    "dedup_tol_arcsec": DEDUP_TOL,
    "refine_sigma_min": REFINE_SIGMA,
    "refine_neighbour_spacing_h": REFINE_NEIGHBOUR_H,
    "refine_factor": REFINE_FACTOR,
    "refine_half_width_h": REFINE_HALF_WIDTH_H,
    "refine_depth": REFINE_DEPTH,
    "refine_max_boxes": REFINE_MAX_BOXES,
    "ray_tracing": "Tracer.traced_grid_2d_list_from (NumPy fp64), no triangle solver",
}


# ---------------------------------------------------------------------------
# Lens equation
# ---------------------------------------------------------------------------


class LensEquation:
    """``r(theta) = traced_plane(theta) - beta`` for one tracer, many source planes."""

    def __init__(self, tracer, planes=None):
        import autolens as al

        self._al = al
        self.tracer = tracer
        self.n_evaluations = 0
        # Planes traced together whenever a full lattice is sampled, so several sources
        # (one per plane) share one multi-plane ray-trace of the lattice.
        self.planes = None if planes is None else sorted({int(p) for p in planes})
        self._lattice_cache = {}

    def lattice_residual(self, key, points, plane_index: int, beta) -> np.ndarray:
        """Residual on a full lattice, the ray-trace cached per lattice ``key`` for every plane."""
        planes = sorted(set(self.planes or []) | {int(plane_index)})
        cached = self._lattice_cache.get(key)
        if cached is None or int(plane_index) not in cached:
            self._lattice_cache[key] = cached = self.traced(points, planes)
        return cached[int(plane_index)] - np.asarray(beta, dtype=float)

    def clear_cache(self):
        self._lattice_cache.clear()

    def traced(self, points: np.ndarray, plane_indices) -> dict[int, np.ndarray]:
        """Traced positions of ``points`` (N, 2) at every plane in ``plane_indices``."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        plane_indices = sorted({int(p) for p in plane_indices})
        out = {p: np.empty_like(points) for p in plane_indices}
        limit = max(plane_indices)
        for start in range(0, points.shape[0], CHUNK):
            block = points[start : start + CHUNK]
            grids = self.tracer.traced_grid_2d_list_from(
                grid=self._al.Grid2DIrregular(block), plane_index_limit=limit
            )
            for p in plane_indices:
                out[p][start : start + CHUNK] = np.asarray(grids[p], dtype=float).reshape(-1, 2)
        self.n_evaluations += points.shape[0]
        return out

    def residual(self, points, plane_index: int, beta) -> np.ndarray:
        return self.traced(points, [plane_index])[plane_index] - np.asarray(beta, dtype=float)

    def jacobian(self, points, plane_index: int) -> np.ndarray:
        """``A = d traced / d theta`` (N, 2, 2) by central differences (step ``FD_STEP``)."""
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        n = p.shape[0]
        dy = np.array([FD_STEP, 0.0])
        dx = np.array([0.0, FD_STEP])
        stencil = np.concatenate([p + dy, p - dy, p + dx, p - dx])
        t = self.traced(stencil, [plane_index])[plane_index]
        col_y = (t[:n] - t[n : 2 * n]) / (2 * FD_STEP)
        col_x = (t[2 * n : 3 * n] - t[3 * n :]) / (2 * FD_STEP)
        return np.stack([col_y, col_x], axis=-1)  # [:, i, j] = d traced_i / d theta_j


# ---------------------------------------------------------------------------
# Seeds and Newton
# ---------------------------------------------------------------------------


def _lattice(y_min, y_max, x_min, x_max, h):
    ys = np.arange(y_min, y_max + 0.5 * h, h)
    xs = np.arange(x_min, x_max + 0.5 * h, h)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return ys, xs, np.stack([yy.ravel(), xx.ravel()], axis=1)


def _seeds(ys, xs, r):
    """Sign-change cell centres + local minima of |r| on an (ny, nx, 2) residual lattice."""
    ny, nx = len(ys), len(xs)
    r = r.reshape(ny, nx, 2)
    seeds = []
    n_sign = 0
    if ny > 1 and nx > 1:
        corners = np.stack([r[:-1, :-1], r[1:, :-1], r[:-1, 1:], r[1:, 1:]])  # (4, ny-1, nx-1, 2)
        lo = corners.min(axis=0)
        hi = corners.max(axis=0)
        cells = np.all((lo <= 0.0) & (hi >= 0.0), axis=-1)
        iy, ix = np.nonzero(cells)
        n_sign = int(iy.size)
        if iy.size:
            seeds.append(
                np.stack([0.5 * (ys[iy] + ys[iy + 1]), 0.5 * (xs[ix] + xs[ix + 1])], axis=1)
            )
    mag = np.hypot(r[..., 0], r[..., 1])
    minima = np.zeros(0, dtype=int)
    min_values = np.zeros(0)
    if ny > 2 and nx > 2:
        centre = mag[1:-1, 1:-1]
        is_min = np.ones_like(centre, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                is_min &= centre <= mag[1 + dy : ny - 1 + dy, 1 + dx : nx - 1 + dx]
        iy, ix = np.nonzero(is_min)
        iy, ix = iy + 1, ix + 1
        values = mag[iy, ix]
        order = np.argsort(values)
        truncated = order.size > MAX_LOCAL_MINIMA
        order = order[:MAX_LOCAL_MINIMA]
        iy, ix = iy[order], ix[order]
        min_values = values[order]
        minima = np.stack([ys[iy], xs[ix]], axis=1)
    else:
        truncated = False
    sign_seeds = np.concatenate(seeds) if seeds else np.zeros((0, 2))
    return sign_seeds, minima.reshape(-1, 2), min_values, n_sign, bool(truncated)


def _newton(eq: LensEquation, plane_index, beta, seeds, max_step):
    """Batched damped Newton from every seed. Returns (positions, |r|, converged mask)."""
    x = np.asarray(seeds, dtype=float).reshape(-1, 2).copy()
    if not x.size:
        return x, np.zeros(0), np.zeros(0, dtype=bool)
    r = eq.residual(x, plane_index, beta)
    n = np.hypot(r[:, 0], r[:, 1])
    active = np.isfinite(n)
    for _ in range(NEWTON_MAX_ITER):
        idx = np.nonzero(active)[0]
        if not idx.size:
            break
        J = eq.jacobian(x[idx], plane_index)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        ok = np.isfinite(det) & (np.abs(det) > 1e-300)
        inv = np.zeros_like(J)
        inv[ok, 0, 0] = J[ok, 1, 1] / det[ok]
        inv[ok, 1, 1] = J[ok, 0, 0] / det[ok]
        inv[ok, 0, 1] = -J[ok, 0, 1] / det[ok]
        inv[ok, 1, 0] = -J[ok, 1, 0] / det[ok]
        dx = -np.einsum("nij,nj->ni", inv, r[idx])
        length = np.hypot(dx[:, 0], dx[:, 1])
        scale = np.where(length > max_step, max_step / np.maximum(length, 1e-300), 1.0)
        dx *= scale[:, None]
        active[idx[~ok]] = False
        # Backtracking line search, every step length of every seed in ONE ray-trace call
        # (a call's fixed cost dominates at cluster scale: 191 profiles per call).
        lams = 2.0 ** -np.arange(LINE_SEARCH_STEPS)
        sel = np.nonzero(ok)[0]
        new_x = x[idx].copy()
        new_r = r[idx].copy()
        new_n = n[idx].copy()
        pending = ok.copy()
        if sel.size:
            trial = x[idx[sel], None, :] + lams[None, :, None] * dx[sel, None, :]
            tr = eq.residual(trial.reshape(-1, 2), plane_index, beta).reshape(
                sel.size, lams.size, 2
            )
            tn = np.hypot(tr[..., 0], tr[..., 1])
            better = np.isfinite(tn) & (tn < n[idx[sel], None])
            has = better.any(axis=1)
            first = np.argmax(better, axis=1)
            acc = sel[has]
            j = first[has]
            new_x[acc] = trial[has, j]
            new_r[acc] = tr[has, j]
            new_n[acc] = tn[has, j]
            pending[acc] = False
        stalled = pending  # no decrease found along the Newton direction (round-off)
        x[idx] = new_x
        r[idx] = new_r
        n[idx] = new_n
        active[idx[stalled]] = False
        active[idx[length * scale < NEWTON_FLOOR_STEP]] = False
    conv = n < ACCEPT_RESIDUAL
    cidx = np.nonzero(conv)[0]
    if cidx.size:
        J = eq.jacobian(x[cidx], plane_index)
        with np.errstate(all="ignore"):
            step = np.array(
                [
                    np.linalg.norm(np.linalg.lstsq(J[k], r[c], rcond=None)[0])
                    for k, c in enumerate(cidx)
                ]
            )
        conv[cidx[~(step < CERTIFY_STEP)]] = False
    return x, n, conv


def _dedup(points, residuals, tol=DEDUP_TOL):
    order = np.argsort(residuals)
    kept = []
    for i in order:
        if all(np.hypot(*(points[i] - points[j])) > tol for j in kept):
            kept.append(i)
    return points[kept].reshape(-1, 2), residuals[kept]


def _describe(eq: LensEquation, plane_index, beta, roots):
    """Residual, A, signed magnification, parity and singular values at each root."""
    if not len(roots):
        return []
    r = eq.residual(roots, plane_index, beta)
    J = eq.jacobian(roots, plane_index)
    out = []
    for k, p in enumerate(roots):
        A = J[k]
        det = float(np.linalg.det(A))
        sv = np.linalg.svd(A, compute_uv=False)
        mu = 1.0 / det if det != 0.0 else float("inf")
        with np.errstate(all="ignore"):
            correction = float(np.linalg.norm(np.linalg.lstsq(A, r[k], rcond=None)[0]))
        out.append(
            {
                "position": [float(p[0]), float(p[1])],
                "residual": float(np.hypot(*r[k])),
                "newton_correction": correction,
                "det_A": det,
                "magnification": mu,
                "parity": int(np.sign(det)),
                "singular_values": [float(sv[0]), float(sv[1])],
                "finite_magnification": bool(np.isfinite(mu)),
            }
        )
    return out


def _search(eq, plane_index, beta, box, h, refine=True):
    """One complete search at lattice spacing ``h``: roots (M, 2) + diagnostics."""
    y_min, y_max, x_min, x_max = box
    ys, xs, pts = _lattice(y_min, y_max, x_min, x_max, h)
    r = eq.lattice_residual((tuple(box), float(h)), pts, plane_index, beta)
    sign_seeds, min_seeds, min_values, n_sign, truncated = _seeds(ys, xs, r)
    seeds = np.concatenate([sign_seeds, min_seeds]) if min_seeds.size else sign_seeds
    x, n, conv = _newton(eq, plane_index, beta, seeds, max_step=4.0 * h)
    roots, res = _dedup(x[conv], n[conv])
    # Failed local-minimum seeds with a small residual: refine around them too.
    n_sign_seeds = sign_seeds.shape[0]
    failed_min = [
        seeds[i]
        for i in range(n_sign_seeds, seeds.shape[0])
        if not conv[i] and min_values[i - n_sign_seeds] < h
    ]
    diag = {
        "h": h,
        "box": list(box),
        "lattice_points": int(pts.shape[0]),
        "sign_change_seeds": int(n_sign),
        "local_minimum_seeds": int(min_seeds.shape[0]),
        "local_minima_truncated": truncated,
        "newton_converged": int(conv.sum()),
        "newton_failed": int((~conv).sum()),
        "refine_boxes": 0,
        "refine_new_roots": 0,
        "refine_capped": False,
    }
    if not refine:
        return roots, diag
    done = set()
    queue = [(p, h, 0, "failed_seed") for p in failed_min]

    def enqueue_from(points, spacing, depth):
        if not len(points):
            return
        J = eq.jacobian(points, plane_index)
        sv_min = np.linalg.svd(J, compute_uv=False)[:, -1]
        for k, p in enumerate(points):
            d = np.hypot(*(roots - p).T)
            d = d[d > DEDUP_TOL]
            near = d.size and d.min() < REFINE_NEIGHBOUR_H * spacing
            # The threshold shrinks with depth as the lattice does: a pair / triplet merging
            # at separation d has sigma_min ~ d, so depth k only chases sigma_min < 0.3 / 16^k.
            critical = sv_min[k] < REFINE_SIGMA / REFINE_FACTOR**depth
            if critical or near:
                queue.append((p, spacing, depth, "near_critical" if critical else "close_pair"))

    enqueue_from(roots, h, 0)
    while queue:
        p, spacing, depth, _why = queue.pop(0)
        if depth >= REFINE_DEPTH:
            continue
        key = (round(float(p[0]) / spacing), round(float(p[1]) / spacing), depth)
        if key in done:
            continue
        if diag["refine_boxes"] >= REFINE_MAX_BOXES:
            diag["refine_capped"] = True
            break
        done.add(key)
        diag["refine_boxes"] += 1
        fine = spacing / REFINE_FACTOR
        half = REFINE_HALF_WIDTH_H * spacing
        sub_box = (p[0] - half, p[0] + half, p[1] - half, p[1] + half)
        fys, fxs, fpts = _lattice(*sub_box, fine)
        fr = eq.residual(fpts, plane_index, beta)
        s_sign, s_min, _s_val, _n, _t = _seeds(fys, fxs, fr)
        s = np.concatenate([s_sign, s_min]) if s_min.size else s_sign
        fx, fn, fconv = _newton(eq, plane_index, beta, s, max_step=4.0 * fine)
        new = []
        for q, qn in zip(fx[fconv], fn[fconv]):
            if not len(roots) or np.hypot(*(roots - q).T).min() > DEDUP_TOL:
                roots = np.concatenate([roots.reshape(-1, 2), q[None]])
                res = np.append(res, qn)
                new.append(q)
        diag["refine_new_roots"] += len(new)
        found = np.asarray(new + [p]).reshape(-1, 2)
        enqueue_from(found[np.isfinite(found).all(axis=1)], fine, depth + 1)
    return roots.reshape(-1, 2), diag


def _match(a, b):
    """Greedy nearest matching of root sets: (pairs (i, j, d), unmatched_a, unmatched_b)."""
    a = np.asarray(a).reshape(-1, 2)
    b = np.asarray(b).reshape(-1, 2)
    if not a.size or not b.size:
        return [], list(range(len(a))), list(range(len(b)))
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    pairs, used_a, used_b = [], set(), set()
    for flat in np.argsort(d, axis=None):
        i, j = np.unravel_index(flat, d.shape)
        if i in used_a or j in used_b:
            continue
        pairs.append((int(i), int(j), float(d[i, j])))
        used_a.add(i)
        used_b.add(j)
    return (
        pairs,
        [i for i in range(len(a)) if i not in used_a],
        [j for j in range(len(b)) if j not in used_b],
    )


def solve(eq: LensEquation, plane_index: int, beta, box, h: float, *, stability=True) -> dict:
    """All lens-equation roots for one source plane, with the acceptance tests applied."""
    t0 = time.perf_counter()
    coarse, diag_c = _search(eq, plane_index, beta, box, h)
    out = {"beta": [float(beta[0]), float(beta[1])], "plane_index": int(plane_index)}
    if stability:
        fine, diag_f = _search(eq, plane_index, beta, box, 0.5 * h)
        pairs, only_coarse, only_fine = _match(coarse, fine)
        moved = {j: dd for _i, j, dd in pairs}
        stable_flags = [(j in moved and moved[j] < STABILITY_TOL) for j in range(fine.shape[0])]
        roots = fine
        out["stability"] = {
            "h_coarse": h,
            "h_fine": 0.5 * h,
            "n_coarse": int(coarse.shape[0]),
            "n_fine": int(fine.shape[0]),
            "max_shift": max((dd for *_, dd in pairs), default=0.0),
            "only_in_coarse": [coarse[i].tolist() for i in only_coarse],
            "only_in_fine": [fine[j].tolist() for j in only_fine],
            "stable": bool(
                not only_coarse and not only_fine and all(dd < STABILITY_TOL for *_, dd in pairs)
            ),
        }
        out["search"] = {"coarse": diag_c, "fine": diag_f}
    else:
        roots = coarse
        stable_flags = [None] * coarse.shape[0]
        out["stability"] = {"stable": None, "h_coarse": h, "note": "stability pass disabled"}
        out["search"] = {"coarse": diag_c}
    described = _describe(eq, plane_index, beta, roots)
    for d, s in zip(described, stable_flags):
        d["stable"] = s
        d["accepted"] = bool(
            d["residual"] < ACCEPT_RESIDUAL
            and d["newton_correction"] < CERTIFY_STEP
            and d["finite_magnification"]
            and s is not False
        )
    out["roots"] = described
    out["n_accepted"] = int(sum(d["accepted"] for d in described))
    out["parity_sum_accepted"] = int(sum(d["parity"] for d in described if d["accepted"]))
    out["wall_s"] = float(time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

# Vendored from autolens_workspace_test/scripts/point_source/solver/error_audit.py (FIXTURES)
# and the six source positions of duplicate_policy_evidence.json "references" (1c/1d).
CALIBRATION_FIXTURES = {
    "analytic_quad": {
        "einstein_radius": 1.6,
        "shear": 0.1,
        "ell_comps": [0.0, 0.0],
    },
    "near_caustic": {
        "einstein_radius": 1.6,
        "ell_comps": [0.05263157894736841, 3.2227547345982974e-18],
    },
}
CALIBRATION_CASES = [
    ("analytic_quad", "boundary", [0.0, 0.0]),
    ("analytic_quad", "offset_plus", [1e-05, 2e-05]),
    ("analytic_quad", "offset_minus", [-1e-05, -2e-05]),
    ("near_caustic", "cusp_0.001", [0.08178909793643739, 0.08178909793643739]),
    ("near_caustic", "cusp_1e-05", [0.08187015019565369, 0.08187015019565369]),
    ("near_caustic", "cusp_1e-07", [0.08187096071824584, 0.08187096071824584]),
]
CALIBRATION_BOX = (-3.0, 3.0, -3.0, 3.0)
CALIBRATION_H = 0.02


def calibration_tracer(fixture):
    import autolens as al

    profiles = {
        "mass": al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=fixture["einstein_radius"],
            ell_comps=tuple(fixture["ell_comps"]),
        )
    }
    if "shear" in fixture:
        profiles["shear"] = al.mp.ExternalShear(gamma_1=fixture["shear"], gamma_2=0.0)
    return al.Tracer(galaxies=[al.Galaxy(redshift=0.5, **profiles), al.Galaxy(redshift=1.0)])


def angular_roots(tracer, fixture, beta, n):
    """Vendored verbatim in method from duplicate_policy.py (1c): isothermal + linear shear only.

    Eliminates the radius from beta = r (u - shear(u)) - alpha_mass(u) and brackets
    the angular roots (Brent). A fixture-specific bracketing oracle, independent of
    both the triangle solver and of this module's Newton search.
    """
    import autolens as al
    from scipy.optimize import brentq

    shear = fixture.get("shear", 0.0)

    def deflections(points):
        return np.asarray(tracer.deflections_yx_2d_from(grid=al.Grid2DIrregular(points)))

    def equation(angles):
        u = np.column_stack((np.sin(angles), np.cos(angles)))
        linear = u * np.array([-shear, shear])
        v = u - linear
        w = np.asarray(beta) + deflections(u) - linear
        return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

    fine = np.geomspace(1e-7, 0.1, 250)
    special = np.concatenate(
        [a + np.r_[-fine, 0.0, fine] for a in np.arange(4) * np.pi / 2 + np.pi / 4]
    )
    angles = np.unique(np.r_[np.linspace(0, 2 * np.pi, n + 1), special])
    values = equation(angles)
    solutions = []
    for i in range(len(angles) - 1):
        if values[i] == 0:
            solutions.append(angles[i])
        elif values[i] * values[i + 1] < 0:
            solutions.append(
                brentq(lambda a: equation(np.array([a]))[0], angles[i], angles[i + 1], xtol=5e-15)
            )
    positions = []
    for a in solutions:
        u = np.array([np.sin(a), np.cos(a)])
        linear = u * np.array([-shear, shear])
        v = u - linear
        w = beta + deflections(u[None])[0] - linear
        radius = np.dot(v, w) / np.dot(v, v)
        p = radius * u
        if radius > 0 and all(np.linalg.norm(p - q) > 1e-6 for q in positions):
            positions.append(p)
    return np.array(sorted(positions, key=lambda p: tuple(p)))


def _workspace_test_solver_dir() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (
            parent / "autolens_workspace_test",
            parent.parent / "autolens_workspace_test" if parent.parent != parent else None,
        ):
            if (
                cand is not None
                and (cand / "scripts/point_source/solver/duplicate_policy.py").exists()
            ):
                return cand / "scripts/point_source/solver"
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upstream(solver_dir: Path | None):
    """The upstream angular_roots function + committed evidence, when the checkout exists."""
    if solver_dir is None:
        return None, None, {"available": False}
    info = {
        "available": True,
        "path": str(solver_dir),
        "duplicate_policy_sha256": _sha256(solver_dir / "duplicate_policy.py"),
        "error_audit_sha256": _sha256(solver_dir / "error_audit.py"),
    }
    evidence_path = solver_dir / "duplicate_policy_evidence.json"
    evidence = None
    if evidence_path.exists():
        info["evidence_sha256"] = _sha256(evidence_path)
        evidence = json.loads(evidence_path.read_text())["references"]
    fn = None
    try:
        sys.path.insert(0, str(solver_dir))
        spec = importlib.util.spec_from_file_location(
            "_upstream_duplicate_policy", solver_dir / "duplicate_policy.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = module.angular_roots
    except Exception as exc:  # noqa: BLE001
        info["import_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sys.path and sys.path[0] == str(solver_dir):
            sys.path.pop(0)
    return fn, evidence, info


def _max_match_error(found, reference):
    pairs, ua, ub = _match(found, reference)
    if ua or ub:
        return None
    return max((d for *_, d in pairs), default=0.0)


def calibrate(h: float = CALIBRATION_H, box=CALIBRATION_BOX) -> dict:
    """Reproduce the 1a analytic quad and the 1c/1d angular_roots fixture to < 1e-6"."""
    t0 = time.perf_counter()
    upstream_fn, evidence, upstream_info = _upstream(_workspace_test_solver_dir())
    cases = []
    for fixture_name, case, beta in CALIBRATION_CASES:
        fixture = CALIBRATION_FIXTURES[fixture_name]
        tracer = calibration_tracer(fixture)
        eq = LensEquation(tracer)
        result = solve(eq, 1, np.asarray(beta), box, h)
        found = np.asarray([r["position"] for r in result["roots"] if r["accepted"]]).reshape(-1, 2)
        references = {}
        if fixture_name == "analytic_quad" and case == "boundary":
            b, g = fixture["einstein_radius"], fixture["shear"]
            references["closed_form"] = np.array(
                [[b / (1 + g), 0], [-b / (1 + g), 0], [0, b / (1 - g)], [0, -b / (1 - g)]]
            )
        references["angular_roots_vendored"] = angular_roots(
            tracer, fixture, np.asarray(beta), 8192
        )
        if upstream_fn is not None:
            references["angular_roots_upstream"] = np.asarray(
                upstream_fn(tracer, fixture, np.asarray(beta), 8192)
            )
        if evidence is not None:
            references["evidence_json"] = np.asarray(evidence[fixture_name][case]["positions"])
        errors = {k: _max_match_error(found, v) for k, v in references.items()}
        passed = bool(
            found.shape[0] == 4
            and result["stability"]["stable"]
            and all(e is not None and e < CALIBRATION_TOL for e in errors.values())
        )
        d = np.hypot(found[:, None, 0] - found[None, :, 0], found[:, None, 1] - found[None, :, 1])
        np.fill_diagonal(d, np.inf)
        cases.append(
            {
                "fixture": fixture_name,
                "case": case,
                "beta": beta,
                "n_oracle_roots": int(found.shape[0]),
                "n_reference_roots": {k: int(len(v)) for k, v in references.items()},
                "max_position_error_vs": errors,
                "oracle_max_residual": max(
                    (r["residual"] for r in result["roots"] if r["accepted"]), default=None
                ),
                "oracle_min_separation": float(d.min()) if found.shape[0] > 1 else None,
                "oracle_min_singular_value": min(
                    (r["singular_values"][1] for r in result["roots"] if r["accepted"]),
                    default=None,
                ),
                "stability": {
                    k: v
                    for k, v in result["stability"].items()
                    if k not in ("only_in_coarse", "only_in_fine")
                },
                "refine_boxes": {k: v["refine_boxes"] for k, v in result["search"].items()},
                "oracle_positions": found.tolist(),
                "wall_s": result["wall_s"],
                "pass": passed,
            }
        )
    return {
        "oracle_version": ORACLE_VERSION,
        "parameters": PARAMETERS,
        "h": h,
        "box": list(box),
        "tolerance_arcsec": CALIBRATION_TOL,
        "upstream": upstream_info,
        "cases": cases,
        "max_error_over_cases": max(
            (e for c in cases for e in c["max_position_error_vs"].values() if e is not None),
            default=None,
        ),
        "pass": all(c["pass"] for c in cases),
        "wall_s": float(time.perf_counter() - t0),
    }


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--h", type=float, default=CALIBRATION_H)
    parser.add_argument("--output", default=None, help="write the calibration JSON here")
    args = parser.parse_args(argv)
    if not args.calibrate:
        parser.error("only --calibrate is a CLI mode; the sweep imports this module")
    result = calibrate(h=args.h)
    for c in result["cases"]:
        errs = ", ".join(
            f"{k}={'n/a' if v is None else f'{v:.2e}'}"
            for k, v in c["max_position_error_vs"].items()
        )
        print(
            f"{c['fixture']:<14} {c['case']:<13} roots {c['n_oracle_roots']} "
            f"minsep {c['oracle_min_separation']:.3e} maxres {c['oracle_max_residual']:.1e} "
            f"stable {c['stability']['stable']} | {errs} | pass {c['pass']}"
        )
    print(
        f"calibration pass: {result['pass']}  max error {result['max_error_over_cases']:.3e}  "
        f"upstream available: {result['upstream']['available']}  wall {result['wall_s']:.1f} s"
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(_main())
