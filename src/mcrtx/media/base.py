"""The medium interface (L2 contract).

A medium is a spherically symmetric background the solvers see only through pure,
differentiable functions of the dimensionless radius ``r`` (in units of the
reference length). Any model satisfying this :class:`Medium` protocol — the
analytic :class:`~mcrtx.media.BetaLawWind`, a :class:`~mcrtx.media.TabulatedWind`
from a dynamics solve, later a 2D/3D grid — plugs into the solvers unchanged
(design decisions D2/D5).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Medium(Protocol):
    """Pure, differentiable functions of dimensionless radius the solvers consume."""

    def v(self, r: Array) -> Array:
        """Outflow speed at ``r`` in units of ``v_inf``."""
        ...

    def dv_dr(self, r: Array) -> Array:
        """Radial velocity gradient ``dv/dr``."""
        ...

    def rho(self, r: Array) -> Array:
        """Density at ``r`` in reference-density units."""
        ...

    def dilution(self, r: Array) -> Array:
        """Geometric dilution factor ``W(r)`` of the stellar core."""
        ...
