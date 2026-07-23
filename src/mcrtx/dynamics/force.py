"""Radiative acceleration from spectral lines — the seed of the dynamics track.

The Sobolev line force per unit mass, in the radial-streaming (point-star)
approximation, is

    g_rad(r) = force_scale * |dv/dr| / (rho r^2) * sum_lines (1 - e^{-tau_S,line}),

where ``tau_S`` is the *radial* Sobolev optical depth of each line. It reduces to
the optically-thin force (`∝ 1/r^2`, every atom absorbs) for weak lines and to
the CAK optically-thick force (`∝ (dv/dr)/(rho r^2)`, saturated) for strong ones.
This is the *general* force — CAK's force multiplier `M(t) = k t^{-alpha}` is one
analytic fit to the line-strength distribution, not needed here.

It is the quantity a self-consistent wind feeds into its momentum equation; today
it is a differentiable diagnostic. ``force_scale`` bundles the physical constants
(`F_0 nu_0 v_inf / (R_* rho_ref c^2)`), analogous to ``tau_scale``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from mcrtx.media import Medium
from mcrtx.physics import sobolev_tau
from mcrtx.physics.source import LineData

__all__ = ["sobolev_line_force"]


def sobolev_line_force(medium: Medium, lines: Sequence[LineData], r: Array, *, force_scale: float = 1.0) -> Array:
    """Radiative acceleration ``g_rad(r)`` from a line list (Sobolev, radial-streaming).

    Args:
        medium: The wind medium (analytic or tabulated).
        lines: The driving lines; each contributes ``1 - e^{-tau_S}``.
        r: Radii at which to evaluate the force (units of the reference length).
        force_scale: Overall dimensionless normalisation (all lines share the
            continuum here; per-line flux weighting is a later refinement).

    Returns:
        The radiative acceleration at each ``r``, same shape as ``r``.
    """
    dv_dr = medium.dv_dr(r)
    rho = medium.rho(r)
    v_over_r = medium.v(r) / r
    radial = jnp.ones_like(r)  # mu = 1: the radial Sobolev optical depth
    saturation = jnp.zeros_like(r)
    for line in lines:
        n_l = rho * r**line.n_l_exponent
        tau_s = sobolev_tau(line.tau_scale, n_l, radial, v_over_r, dv_dr)
        saturation = saturation + (1.0 - jnp.exp(-tau_s))
    return force_scale * jnp.abs(dv_dr) / (rho * r**2) * saturation
