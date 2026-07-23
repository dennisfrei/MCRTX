"""Atomic line data to kernel line data (unit boundary, D4).

Atomic databases (NIST, VALD) publish wavelengths and oscillator strengths;
the solvers consume the dimensionless :class:`~mcrtx.physics.source.LineData`.
:func:`build_line_list` is the one-way bridge, so callers never hand-roll a
``tau_scale``.

The Sobolev optical depth of a resonance line is

    tau = (pi e^2 / m_e c) f_lu lambda_0 n_l / |dv_l/ds|,

which becomes ``tau_scale * n_l / |dv_l/ds|`` in kernel units once ``n_l`` is
measured in ``scales.density`` and the gradient in ``scales.velocity /
scales.length``. Hence ``tau_scale`` carries a factor ``scales.time``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import unxt as u
from astropy import constants as const
from jax import Array
from unxt import AbstractQuantity, Quantity

from mcrtx.physics.source import LineData
from mcrtx.units import ReferenceScales

__all__ = ["build_line_list"]

# astropy builds its constants at import time, so they are invisible to the static
# checker; their values are asserted in lines_test.py.
_E_ESU = const.e.esu  # ty: ignore[unresolved-attribute]
_M_E = const.m_e.cgs  # ty: ignore[unresolved-attribute]
_C = const.c.cgs  # ty: ignore[unresolved-attribute]

# pi e^2 / (m_e c), the classical oscillator constant, in cm^2 / s.
_OSCILLATOR_CONSTANT = float((np.pi * _E_ESU**2 / (_M_E * _C)).to_value("cm2 / s"))
_SPEED_OF_LIGHT = float(_C.value)


def _length_unit() -> Any:
    # unxt units are dynamically typed at this boundary; keep the Any here.
    return u.unit("cm")


def _strip_length(quantity: AbstractQuantity, name: str) -> np.ndarray:
    unit = _length_unit()
    expected = u.dimension_of(Quantity(1.0, unit))
    if u.dimension_of(quantity) != expected:
        raise ValueError(f"{name} has dimension {u.dimension_of(quantity)}, expected {expected}")
    values = np.atleast_1d(np.asarray(quantity.ustrip(unit), dtype=float))
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must be positive, got {values.min()}")
    return values


def build_line_list(
    wavelengths: AbstractQuantity,
    f_values: Sequence[float] | Array,
    scales: ReferenceScales,
    *,
    reference_wavelength: AbstractQuantity | None = None,
    n_l_exponent: float | Sequence[float] = 0.0,
    stimulated: float | Sequence[float] = 1.0,
    g_lower: float | Sequence[float] = 1.0,
    g_upper: float | Sequence[float] = 3.0,
) -> tuple[Array, list[LineData]]:
    """Convert tabulated atomic data into solver-ready line centres and line data.

    The returned pair plugs straight into
    :func:`~mcrtx.solvers.multiline.multiline_profile_coupled` and
    :func:`~mcrtx.dynamics.force.sobolev_line_force`; input order is preserved,
    so ``centers[i]`` pairs with ``lines[i]``.

    Plasma-dependent NLTE fields (``collision``, ``w_core``, ``boltzmann``) are
    not line data and stay unset; attach them per line with ``LineData._replace``.

    Args:
        wavelengths: Rest wavelengths of the lines (dimension: length), shape ``(N,)``.
        f_values: Absorption oscillator strengths ``f_lu``, shape ``(N,)``.
        scales: Reference scales of the run; supplies the lower-level number
            density and the ``length / velocity`` time scale.
        reference_wavelength: Zero point of the velocity axis; defaults to the
            first entry of ``wavelengths``.
        n_l_exponent: Population power-law tweak, scalar or per line.
        stimulated: Stimulated-emission correction ``1 - g_l n_u / (g_u n_l)``,
            scalar or per line; ``1.0`` is pure absorption.
        g_lower: Lower-level statistical weight, scalar or per line.
        g_upper: Upper-level statistical weight, scalar or per line.

    Returns:
        ``(centers, lines)``: Doppler offsets from ``reference_wavelength`` in
        units of ``scales.velocity``, and the matching dimensionless line data.

    Raises:
        ValueError: If a wavelength has the wrong dimension or is not positive,
            if ``f_values`` does not match ``wavelengths`` in length or is
            negative, or if a per-line field cannot broadcast to ``(N,)``.
    """
    lam = _strip_length(wavelengths, "wavelengths")
    f_lu = np.atleast_1d(np.asarray(f_values, dtype=float))
    if f_lu.shape != lam.shape:
        raise ValueError(f"f_values has shape {f_lu.shape}, expected {lam.shape} to match wavelengths")
    if np.any(f_lu < 0.0):
        raise ValueError(f"f_values must be non-negative, got {f_lu.min()}")

    if reference_wavelength is None:
        lam_ref = lam[0]
    else:
        ref = _strip_length(reference_wavelength, "reference_wavelength")
        if ref.size != 1:
            raise ValueError(f"reference_wavelength must be scalar, got shape {ref.shape}")
        lam_ref = ref[0]

    per_line = {
        "n_l_exponent": n_l_exponent,
        "stimulated": stimulated,
        "g_lower": g_lower,
        "g_upper": g_upper,
    }
    fields: dict[str, np.ndarray] = {}
    for name, value in per_line.items():
        array = np.atleast_1d(np.asarray(value, dtype=float))
        if array.shape not in {(1,), lam.shape}:
            raise ValueError(f"{name} has shape {array.shape}, expected a scalar or {lam.shape}")
        fields[name] = np.broadcast_to(array, lam.shape)

    centers = jnp.asarray((lam - lam_ref) / lam_ref * (_SPEED_OF_LIGHT / scales.velocity))
    tau_scale = _OSCILLATOR_CONSTANT * f_lu * lam * scales.density * scales.time * fields["stimulated"]

    lines = [
        LineData(
            tau_scale=jnp.asarray(tau_scale[i]),
            n_l_exponent=jnp.asarray(fields["n_l_exponent"][i]),
            g_lower=float(fields["g_lower"][i]),
            g_upper=float(fields["g_upper"][i]),
        )
        for i in range(lam.size)
    ]
    return centers, lines
