"""Sobolev optical depth for radial flows (concept sec. 3.3).

The core formula is a purely local, algebraic function — no integration —
which makes it ideal for vectorisation over packets and radii alike.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["projected_velocity_gradient", "sobolev_tau"]


def projected_velocity_gradient(mu: Array, v_over_r: Array | float, dv_dr: Array | float) -> Array:
    """Line-of-sight velocity gradient ``dv_l/ds = mu^2 dv/dr + (1-mu^2) v/r``.

    Strictly positive for a monotonically accelerating radial outflow.
    """
    mu2 = mu * mu
    return mu2 * dv_dr + (1.0 - mu2) * v_over_r


def sobolev_tau(tau_scale: Array, n_l: Array, mu: Array, v_over_r: Array | float, dv_dr: Array | float) -> Array:
    """Direction-dependent Sobolev optical depth.

    ``tau = tau_scale * n_l / |dv_l/ds|`` where ``tau_scale`` bundles the
    atomic constants ``(pi e^2 / m_e c) f_lu lambda_0`` and the stimulated-
    emission correction in dimensionless form. Keeping ``tau_scale`` a single
    differentiable parameter is what makes the M0 gradient acceptance test
    (grad w.r.t. line strength) clean.

    Args:
        tau_scale: Dimensionless line-strength scale (broadcastable).
        n_l: Lower-level population in reference-density units, at the
            resonance point(s).
        mu: Direction cosine at the resonance point(s).
        v_over_r: Local ``v / r`` at the resonance point(s).
        dv_dr: Local radial velocity gradient at the resonance point(s).

    Returns:
        Sobolev optical depth with the broadcast shape of the inputs.
    """
    grad = projected_velocity_gradient(mu, v_over_r, dv_dr)
    return tau_scale * n_l / jnp.abs(grad)
