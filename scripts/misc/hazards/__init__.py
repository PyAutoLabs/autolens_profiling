"""Numerical-hazard profiling framework.

The package records where PyAuto likelihood ingredients are non-smooth,
ill-conditioned, or backend-dependent.  Reusable detectors live here; dataset
specific fixtures belong under ``scripts/<dataset>/<model>/``.
"""

from ._measure import Measurement
from ._record import Finding

__all__ = ["Finding", "Measurement"]
