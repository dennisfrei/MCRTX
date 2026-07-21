"""Spherically symmetric beta-law wind (MVP medium, concept sec. 3.3).

All quantities dimensionless: radii in units of ``R_star``, velocities in
units of ``v_inf``, densities in units of the reference density.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

__all__ = ["BetaLawWind"]


class BetaLawWind(NamedTuple):
    """Beta velocity law ``v(r) = (1 - b / r)**beta`` with ``v(1) = v_phot``.

    Attributes:
        beta: Velocity-law exponent (typically ``0.5 .. 2``).
        v_phot: Photospheric velocity in units of ``v_inf`` (sets ``b``).
        mdot_scale: Dimensionless mass-loss scale fixing ``rho`` via continuity.
    """

    beta: Array
    v_phot: Array
    mdot_scale: Array

    @property
    def b(self) -> Array:
        """Offset parameter enforcing ``v(1) = v_phot``."""
        return 1.0 - self.v_phot ** (1.0 / self.beta)

    def v(self, r: Array) -> Array:
        """Wind speed at radius ``r`` (units of ``v_inf``)."""
        return (1.0 - self.b / r) ** self.beta

    def dv_dr(self, r: Array) -> Array:
        """Radial velocity gradient ``dv/dr``."""
        return self.beta * (1.0 - self.b / r) ** (self.beta - 1.0) * self.b / r**2

    def rho(self, r: Array) -> Array:
        """Density from mass continuity, ``rho ∝ 1 / (r^2 v)``."""
        return self.mdot_scale / (r**2 * self.v(r))

    def dilution(self, r: Array) -> Array:
        """Geometric dilution factor ``W(r) = 0.5 (1 - sqrt(1 - 1/r^2))``."""
        return 0.5 * (1.0 - jnp.sqrt(1.0 - 1.0 / r**2))
