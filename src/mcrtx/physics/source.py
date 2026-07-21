"""Source functions / level populations (exchangeable module, D2).

M0: two-level atom with geometric dilution factor — the classic
escape-probability closure (Sobolev 1960; Castor 1970). The interface is
deliberately not pinned to two levels: an NLTE rate-equation solver
(Sobolev-beta coupled, one batched linear system per shell) replaces this
module in M1 without touching the core API.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

__all__ = ["LineData", "escape_probability"]


class LineData(NamedTuple):
    """Static data of a single resonance line (dimensionless).

    Attributes:
        tau_scale: Line-strength scale entering
            :func:`mcrtx.physics.sobolev.sobolev_tau`.
        n_l_exponent: MVP population model: ``n_l(r) ∝ rho(r)`` scaled by this
            power-law tweak (placeholder until M1 rate equations).
    """

    tau_scale: Array
    n_l_exponent: Array


def escape_probability(tau: Array) -> Array:
    """Sobolev escape probability ``beta(tau) = (1 - exp(-tau)) / tau``.

    Implemented with a Taylor switch at small ``tau`` for numerical safety
    and well-defined gradients at ``tau -> 0``.

    Args:
        tau: Sobolev optical depth (broadcastable).

    Returns:
        Escape probability with the broadcast shape of ``tau``.
    """
    small = tau < 1e-4
    safe = jnp.where(small, 1.0, tau)
    beta = (1.0 - jnp.exp(-safe)) / safe
    return jnp.where(small, 1.0 - 0.5 * tau, beta)
