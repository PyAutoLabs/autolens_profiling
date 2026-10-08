"""Paired whole-round bootstrap for the A/B cells' reported intervals (#362, fix phase 4).

Why this module exists
----------------------

``results/notes/timing_noise_audit_2026_10.md`` (rows C6, C8, C9 and the
"reported CIs" of the estimator table) found every point-source A/B cell
bootstrapping its median ratios by resampling **individual calls iid, unpaired**
— ``np.median(rng.choice(num, num.size)) / np.median(rng.choice(den,
den.size))``. The cells time routes in interleaved round-robin rounds
(``--calls`` consecutive calls of one route per round, start rotated each
round). Calls inside one round share that round's state (frequency step,
neighbour load, cache residency), so they are not independent, and the iid
interval is too narrow; and the routes are paired by round, which the unpaired
resampling throws away.

This module resamples **whole rounds, with the same round indices for every
route** (a paired cluster bootstrap). The number of rounds is the effective
sample size and is recorded as ``effective_n``. The interval estimator is
otherwise unchanged (median of the pooled calls of the drawn rounds, 90 %
percentile interval, a fixed seed), so the keys a cell already writes keep their
meaning. Every cell imports :func:`round_median_ratio` /
:func:`round_median_saving` from here; ``scripts/misc/test/test_round_bootstrap.py``
asserts the same function objects.

Limits
------

- A percentile bootstrap over ``n_rounds`` clusters is approximate; with fewer
  than :data:`likelihood_breakdown.ab_verdict.MIN_AB_ROUNDS` rounds the
  verdicts reading it are INCONCLUSIVE anyway.
- Rounds are treated as exchangeable. A slow drift across rounds is still
  absorbed into the interval as scatter (rounds rotate the route order, so it is
  not charged to one route), not modelled.
"""

from __future__ import annotations

import numpy as np

#: The cells' bootstrap size and confidence (unchanged from their iid versions).
ROUND_BOOTSTRAP_SAMPLES = 2000
ROUND_BOOTSTRAP_CONFIDENCE = 0.90
RESAMPLING = "paired whole rounds"


def as_rounds(calls, n_rounds: int) -> np.ndarray:
    """``calls`` (in round order, ``n_calls`` per round) as an ``(n_rounds, n_calls)`` array."""
    arr = np.asarray(calls, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] != int(n_rounds):
            raise ValueError(f"expected {n_rounds} rounds, got {arr.shape[0]}")
        return arr
    n_rounds = int(n_rounds)
    if n_rounds < 1 or arr.ndim != 1 or arr.size == 0 or arr.size % n_rounds:
        raise ValueError(f"{arr.size} calls do not split into {n_rounds} equal rounds")
    return arr.reshape(n_rounds, arr.size // n_rounds)


def _draws(rounds_a: np.ndarray, rounds_b: np.ndarray, seed: int, samples: int):
    """Medians of both arms over the same resampled round indices, per draw."""
    n_rounds = rounds_a.shape[0]
    if rounds_b.shape[0] != n_rounds:
        raise ValueError("both arms must have the same number of rounds (they are paired)")
    rng = np.random.default_rng(seed)
    med_a = np.empty(samples)
    med_b = np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, n_rounds, n_rounds)
        med_a[i] = np.median(rounds_a[idx])
        med_b[i] = np.median(rounds_b[idx])
    return med_a, med_b


def _percentiles(boots: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = 50.0 * (1.0 - confidence)
    return float(np.percentile(boots, tail)), float(np.percentile(boots, 100.0 - tail))


def round_median_ratio(
    numerator,
    denominator,
    *,
    n_rounds: int,
    seed: int,
    samples: int = ROUND_BOOTSTRAP_SAMPLES,
    confidence: float = ROUND_BOOTSTRAP_CONFIDENCE,
    return_boots: bool = False,
):
    """``median(numerator) / median(denominator)`` with a paired round-bootstrap interval.

    Returns the cells' dict (``ratio``, ``ci90_low``, ``ci90_high``,
    ``bootstrap_samples``) plus ``resampling`` and ``effective_n``; with
    ``return_boots`` also the bootstrap ratios.
    """
    num = as_rounds(numerator, n_rounds)
    den = as_rounds(denominator, n_rounds)
    med_num, med_den = _draws(num, den, seed, samples)
    boots = med_num / med_den
    low, high = _percentiles(boots, confidence)
    out = {
        "ratio": float(np.median(num) / np.median(den)),
        "ci90_low": low,
        "ci90_high": high,
        "bootstrap_samples": int(samples),
        "resampling": RESAMPLING,
        "effective_n": int(num.shape[0]),
    }
    return (out, boots) if return_boots else out


def round_median_saving(
    slower,
    faster,
    *,
    n_rounds: int,
    seed: int,
    scale: float = 1.0e3,
    samples: int = ROUND_BOOTSTRAP_SAMPLES,
    confidence: float = ROUND_BOOTSTRAP_CONFIDENCE,
) -> dict:
    """``(median(slower) - median(faster)) * scale`` with a paired round-bootstrap interval.

    ``scale`` converts seconds to the cells' milliseconds by default. Returns the
    cells' dict (``saved_ms``, ``ci90_low``, ``ci90_high``,
    ``bootstrap_samples``) plus ``resampling`` and ``effective_n``.
    """
    slow = as_rounds(slower, n_rounds) * scale
    fast = as_rounds(faster, n_rounds) * scale
    med_slow, med_fast = _draws(slow, fast, seed, samples)
    low, high = _percentiles(med_slow - med_fast, confidence)
    return {
        "saved_ms": float(np.median(slow) - np.median(fast)),
        "ci90_low": low,
        "ci90_high": high,
        "bootstrap_samples": int(samples),
        "resampling": RESAMPLING,
        "effective_n": int(slow.shape[0]),
    }
