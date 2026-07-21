"""Unit boundary (design decision D4).

Units exist only at the API boundary. On kernel entry, all quantities are
non-dimensionalised against a configurable :class:`ReferenceScales` set; the
hot path computes on plain ``jax.Array`` values. On exit, results are
converted back to ``unxt.Quantity`` in the user's chosen unit system.

Kernel contract: every dimensionless field documents its reference scale.
"Dimensionless" never means "convention-free".
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["ReferenceScales"]


class ReferenceScales(NamedTuple):
    """Reference scales defining the dimensionless system of a run.

    All values are given in a coherent unit system (validated at construction
    via unxt) and stored as plain floats for the kernel.

    Attributes:
        length: Reference length, canonically the stellar radius ``R_star``.
        velocity: Reference velocity, canonically the terminal wind speed ``v_inf``.
        frequency: Reference frequency, canonically the line rest frequency ``nu_0``.
        density: Reference number density of the lower level, evaluated at a
            canonical radius (default: ``2 R_star``).
    """

    length: float
    velocity: float
    frequency: float
    density: float

    # TODO(M0): classmethod ``from_quantities`` accepting unxt Quantities,
    #   validating dimensions and returning floats in a coherent system.
    # TODO(M0): presets ``cgs``, ``si``, ``wind_natural``.
