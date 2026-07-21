"""Output accumulators (estimators).

MVP: observer-frame frequency histogram of escaped packet weights
(the emergent line profile). Additionally the mean-intensity estimator
``j_bar`` per radial shell is *logged but unused* in M0 — it is the
prepared coupling point for the NLTE rate equations in M1.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

__all__ = ["Estimators"]


class Estimators(NamedTuple):
    """Accumulated run outputs (dimensionless; converted at the API boundary).

    Attributes:
        spectrum: Escaped weight per observer-frame frequency bin, shape ``(n_nu,)``.
        j_bar: Line mean-intensity estimator per radial shell, shape
            ``(n_shells,)``. Logged in M0, consumed from M1 on.
    """

    spectrum: Array
    j_bar: Array

    # TODO(M0): pure update functions ``deposit_escape(est, packets)`` and
    #   ``deposit_resonance(est, shell_idx, weights)`` (scatter_add based).
