"""Whole-likelihood helper: a log evidence from the numba w-tilde inversion.

PyAutoArray's inversion factory has no route to ``InversionInterferometerNumba``
(the pack lives outside the library and the library is read-only here), so this
module rebuilds the *inversion* from an ordinary ``al.FitInterferometer`` while
reusing everything the fit already built — the tracer, the ray-traced grids, the
Delaunay triangulation and the mapper all arrive via the fit's
``inversion.linear_obj_list``.

``numba_log_evidence_from`` therefore measures the same figure of merit
``FitInterferometer.log_evidence`` measures (same ``fit_util.log_evidence_from``,
same ``fast_chi_squared``, same ``noise_normalization``), with only the curvature
assembly swapped for the recovered numba path.
"""

from __future__ import annotations

import numpy as np
from autoarray.fit import fit_util

from numba_interferometer.inversion import InversionInterferometerNumba
from numba_interferometer.preload import NumbaPreload


def numba_inversion_from(
    fit,
    numba_preload: NumbaPreload = None,
    kernel: str = "reference",
) -> InversionInterferometerNumba:
    """Rebuild ``fit``'s inversion on the numba w-tilde path.

    Parameters
    ----------
    fit
        An ``al.FitInterferometer`` built with ``xp=np`` whose dataset has had
        ``apply_sparse_operator`` called on it.
    numba_preload
        The dataset's :class:`NumbaPreload`; built from the dataset when omitted.
    kernel
        Which curvature kernel assembles ``F`` (see ``InversionInterferometerNumba``).
    """
    inversion = fit.inversion

    if inversion is None:
        raise ValueError(
            "The `FitInterferometer` passed to `numba_inversion_from` has no inversion, so "
            "there is no pixelized reconstruction for the numba w-tilde path to rebuild."
        )

    return InversionInterferometerNumba(
        dataset=inversion.dataset,
        linear_obj_list=inversion.linear_obj_list,
        settings=inversion.settings,
        xp=np,
        numba_preload=numba_preload,
        kernel=kernel,
    )


def numba_log_evidence_from(
    fit,
    numba_preload: NumbaPreload = None,
    kernel: str = "reference",
):
    """Returns ``(log_evidence, inversion)`` for ``fit`` on the numba w-tilde path.

    The five terms are read off the numba inversion exactly as
    ``FitInterferometer.log_evidence`` reads them off the sparse one, so the returned
    value is comparable to ``fit.figure_of_merit`` term for term.
    """
    inversion = numba_inversion_from(fit=fit, numba_preload=numba_preload, kernel=kernel)

    log_evidence = fit_util.log_evidence_from(
        chi_squared=inversion.fast_chi_squared,
        regularization_term=inversion.regularization_term,
        log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
        log_regularization_term=inversion.log_det_regularization_matrix_term,
        noise_normalization=fit.noise_normalization,
    )

    return float(log_evidence), inversion
