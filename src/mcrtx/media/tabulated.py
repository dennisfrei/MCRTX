"""A medium defined by tabulated radial profiles.

Where :class:`~mcrtx.media.BetaLawWind` is analytic, :class:`TabulatedWind`
carries ``v(r)``, ``dv/dr`` and ``rho(r)`` sampled on a radius grid and
interpolates them — the shape a *solved* medium takes (a radiation-driven
dynamics solve, or a hydro grid). It satisfies the same :class:`~mcrtx.media.Medium`
protocol, so the solvers use it interchangeably with the analytic wind.
"""

from __future__ import annotations

from typing import NamedTuple, Self

import jax.numpy as jnp
from jax import Array

from mcrtx.media.base import Medium

__all__ = ["TabulatedWind"]


class TabulatedWind(NamedTuple):
    """Medium interpolated from tabulated radial profiles (log-``r`` grid recommended).

    Radii below/above the grid clamp to the endpoints; ``dilution`` is the exact
    geometric factor, independent of the tabulation.

    Attributes:
        r_grid: Monotonically increasing radii, shape ``(N,)``.
        v_grid: Outflow speed at each radius (units of ``v_inf``).
        dv_grid: Radial velocity gradient at each radius.
        rho_grid: Density at each radius (reference-density units).
    """

    r_grid: Array
    v_grid: Array
    dv_grid: Array
    rho_grid: Array

    @classmethod
    def sample(cls, medium: Medium, r_grid: Array) -> Self:
        """Tabulate any :class:`~mcrtx.media.Medium` onto ``r_grid``."""
        return cls(r_grid, medium.v(r_grid), medium.dv_dr(r_grid), medium.rho(r_grid))

    def v(self, r: Array) -> Array:
        """Outflow speed at ``r`` in units of ``v_inf``."""
        return jnp.interp(r, self.r_grid, self.v_grid)

    def dv_dr(self, r: Array) -> Array:
        """Radial velocity gradient ``dv/dr``."""
        return jnp.interp(r, self.r_grid, self.dv_grid)

    def rho(self, r: Array) -> Array:
        """Density at ``r`` in reference-density units."""
        return jnp.interp(r, self.r_grid, self.rho_grid)

    def dilution(self, r: Array) -> Array:
        """Geometric dilution factor ``W(r) = 0.5 (1 - sqrt(1 - 1/r^2))``."""
        return 0.5 * (1.0 - jnp.sqrt(1.0 - 1.0 / r**2))
