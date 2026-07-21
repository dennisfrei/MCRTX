"""Unit boundary (design decision D4).

Units exist only at the API boundary. On kernel entry, all quantities are
non-dimensionalised against a configurable :class:`ReferenceScales` set; the
hot path computes on plain ``jax.Array`` values. On exit, results are
converted back to ``unxt.Quantity`` in the user's chosen unit system.

Kernel contract: every dimensionless field documents its reference scale.
"Dimensionless" never means "convention-free".
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple, Self

import jax.numpy as jnp
import numpy as np
import unxt as u
from jax import Array
from unxt import AbstractQuantity, Quantity


class Scale(StrEnum):
    """Physical kind of a reference scale (a field name of :class:`ReferenceScales`)."""

    LENGTH = "length"
    VELOCITY = "velocity"
    FREQUENCY = "frequency"
    DENSITY = "density"


def _quantity(value: Any, unit: Any) -> Quantity:
    # unxt units are dynamically typed at this boundary; keep the Any here.
    return Quantity(value, unit)


# Coherent CGS base unit for each scale; magnitudes are stored in these units so
# that dimensionless ratios are independent of the user's input unit system.
_BASE_UNIT: dict[Scale, Any] = {
    Scale.LENGTH: u.unit("cm"),
    Scale.VELOCITY: u.unit("cm / s"),
    Scale.FREQUENCY: u.unit("Hz"),
    Scale.DENSITY: u.unit("cm**-3"),
}

__all__ = ["ReferenceScales", "Scale"]


class ReferenceScales(NamedTuple):
    """Reference scales defining the dimensionless system of a run.

    Magnitudes are stored in coherent CGS base units (validated at construction
    via unxt) as plain floats for the kernel. Non-dimensionalising a physical
    quantity divides its CGS magnitude by the matching scale, so the resulting
    dimensionless value is the same whether the input was given in CGS or SI.

    Attributes:
        length: Reference length in cm, canonically the stellar radius ``R_star``.
        velocity: Reference velocity in cm/s, canonically the terminal wind speed ``v_inf``.
        frequency: Reference frequency in Hz, canonically the line rest frequency ``nu_0``.
        density: Reference number density in cm^-3 of the lower level, evaluated at a
            canonical radius (default: ``2 R_star``).
    """

    length: float
    velocity: float
    frequency: float
    density: float

    @property
    def time(self) -> float:
        """Derived time scale ``length / velocity`` in seconds."""
        return self.length / self.velocity

    @classmethod
    def from_quantities(
        cls,
        length: AbstractQuantity,
        velocity: AbstractQuantity,
        frequency: AbstractQuantity,
        density: AbstractQuantity,
    ) -> Self:
        """Build reference scales from dimensional unxt quantities.

        Each argument is validated against its expected physical dimension and
        converted to the coherent CGS base unit before being stored as a float.

        Args:
            length: Reference length (dimension: length).
            velocity: Reference velocity (dimension: speed).
            frequency: Reference frequency (dimension: frequency).
            density: Reference number density (dimension: number density).

        Returns:
            Reference scales with CGS magnitudes.

        Raises:
            ValueError: If any argument has the wrong dimension or is not positive.
        """
        values = {
            Scale.LENGTH: length,
            Scale.VELOCITY: velocity,
            Scale.FREQUENCY: frequency,
            Scale.DENSITY: density,
        }
        magnitudes: dict[Scale, float] = {}
        for scale, quantity in values.items():
            base = _BASE_UNIT[scale]
            expected = u.dimension_of(_quantity(1.0, base))
            if u.dimension_of(quantity) != expected:
                raise ValueError(f"{scale.value} has dimension {u.dimension_of(quantity)}, expected {expected}")
            magnitude = float(np.asarray(quantity.ustrip(base)).item())
            if magnitude <= 0.0:
                raise ValueError(f"{scale.value} scale must be positive, got {magnitude}")
            magnitudes[scale] = magnitude
        return cls(
            length=magnitudes[Scale.LENGTH],
            velocity=magnitudes[Scale.VELOCITY],
            frequency=magnitudes[Scale.FREQUENCY],
            density=magnitudes[Scale.DENSITY],
        )

    @classmethod
    def cgs(cls) -> Self:
        """Pass-through CGS scales (all unit): dimensionless values equal CGS magnitudes."""
        return cls(length=1.0, velocity=1.0, frequency=1.0, density=1.0)

    @classmethod
    def si(cls) -> Self:
        """Pass-through SI scales: dimensionless values equal SI magnitudes."""
        return cls.from_quantities(
            _quantity(1.0, u.unit("m")),
            _quantity(1.0, u.unit("m / s")),
            _quantity(1.0, u.unit("Hz")),
            _quantity(1.0, u.unit("m**-3")),
        )

    def nondimensionalize(self, quantity: AbstractQuantity, scale: Scale) -> Array:
        """Convert a physical quantity to a dimensionless kernel value.

        Args:
            quantity: The dimensional quantity to convert.
            scale: Which reference scale to divide by.

        Returns:
            The dimensionless magnitude ``quantity / scale``.
        """
        return jnp.asarray(quantity.ustrip(_BASE_UNIT[scale])) / getattr(self, scale.value)

    def redimensionalize(self, value: Array, scale: Scale, unit: str | None = None) -> AbstractQuantity:
        """Convert a dimensionless kernel value back to a physical quantity.

        Args:
            value: The dimensionless magnitude in units of ``scale``.
            scale: Which reference scale the value is expressed in.
            unit: Optional output unit; defaults to the coherent CGS base unit.

        Returns:
            The dimensional quantity, in ``unit`` if given.
        """
        quantity = _quantity(value * getattr(self, scale.value), _BASE_UNIT[scale])
        return quantity.uconvert(unit) if unit is not None else quantity
