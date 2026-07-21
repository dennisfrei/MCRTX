"""MVP scenario: single resonance line in a 1D stellar wind.

The only place where units, defaults and solver selection meet (L5). Physical
``unxt`` quantities enter, are non-dimensionalised against a
:class:`~mcrtx.units.ReferenceScales` built from the wind's own scales, handed
to the chosen solver, and the emergent profile comes back on a physical Doppler
velocity axis. Because everything flows through the reference scales, the
dimensionless core result is identical whether inputs are given in cgs or SI.

M0 keeps ``mdot_scale`` and ``tau_scale`` as dimensionless drivers; deriving
them from a physical mass-loss rate and atomic data is a later refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
import unxt as u
from jax import Array
from unxt import AbstractQuantity, Quantity

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import run_profile, solve_profile
from mcrtx.units import ReferenceScales, Scale

__all__ = [
    "Method",
    "RunConfig",
    "WindLineResult",
    "WindLineScenario",
    "continuity_number_density",
    "sobolev_tau_scale",
    "solve",
]


def _quantity(value: Any, unit: Any) -> Quantity:
    # unxt units are dynamically typed at this boundary; keep the Any here.
    return Quantity(value, unit)


# pi e^2 / (m_e c), the classical integrated-line constant, and the speed of light.
_PI_E2_ME_C = _quantity(0.02654008, u.unit("cm**2 / s"))
_C = _quantity(2.99792458e10, u.unit("cm / s"))


def continuity_number_density(
    mdot: AbstractQuantity,
    r_star: AbstractQuantity,
    v_inf: AbstractQuantity,
    lower_level_per_mass: AbstractQuantity,
) -> AbstractQuantity:
    """Lower-level number-density scale from mass continuity.

    ``n_l(r) = n_0 / (r~^2 v~)`` with ``n_0 = (lower_level_per_mass) Mdot / (4 pi R*^2 v_inf)``,
    so a scenario built with this density has ``mdot_scale = 1``.

    Args:
        mdot: Mass-loss rate (dimension: mass / time).
        r_star: Stellar radius.
        v_inf: Terminal wind speed.
        lower_level_per_mass: Lower-level absorbers per unit mass (dimension: 1 / mass),
            lumping element abundance, ionisation fraction and mean molecular weight.

    Returns:
        The reference number density ``n_0`` in cm^-3.
    """
    # Form the mass density first to avoid a large intermediate under float32.
    rho_mass = mdot / (4.0 * np.pi * r_star**2 * v_inf)
    return (lower_level_per_mass * rho_mass).uconvert(u.unit("cm**-3"))


def sobolev_tau_scale(
    f_lu: float,
    nu_0: AbstractQuantity,
    density_ref: AbstractQuantity,
    r_star: AbstractQuantity,
    v_inf: AbstractQuantity,
    *,
    stimulated: float = 1.0,
) -> float:
    """Dimensionless Sobolev line-strength scale from atomic data.

    ``tau_scale = (pi e^2 / m_e c) f_lu lambda_0 n_ref [stim] (R* / v_inf)``, the
    constant bundled into :func:`~mcrtx.physics.sobolev.sobolev_tau`.

    Args:
        f_lu: Absorption oscillator strength.
        nu_0: Line rest frequency.
        density_ref: Reference lower-level number density.
        r_star: Stellar radius.
        v_inf: Terminal wind speed.
        stimulated: Stimulated-emission correction ``1 - (g_l n_u)/(g_u n_l)`` (default 1).

    Returns:
        The dimensionless line-strength scale.
    """
    lam_0 = _C / nu_0
    tau = _PI_E2_ME_C * f_lu * lam_0 * density_ref * stimulated * (r_star / v_inf)
    return float(np.asarray(tau.ustrip(u.unit(""))).item())


class Method(StrEnum):
    """Transfer solver to run a scenario with."""

    REFERENCE = "reference"
    MC = "mc"


@dataclass(frozen=True)
class WindLineScenario:
    """Physical description of a 1D wind with a single resonance line.

    Attributes:
        r_star: Stellar radius (reference length).
        v_inf: Terminal wind speed (reference velocity).
        v_phot: Photospheric velocity at ``r = R_star``.
        nu_0: Line rest frequency (reference frequency).
        density_ref: Reference number density of the lower level.
        beta: Velocity-law exponent.
        mdot_scale: Dimensionless mass-loss scale fixing the density.
        tau_scale: Dimensionless line-strength scale.
        n_l_exponent: Radial power-law tweak of the lower-level population.
    """

    r_star: AbstractQuantity
    v_inf: AbstractQuantity
    v_phot: AbstractQuantity
    nu_0: AbstractQuantity
    density_ref: AbstractQuantity
    beta: float
    mdot_scale: float
    tau_scale: float
    n_l_exponent: float = 0.0

    @classmethod
    def from_physical(
        cls,
        *,
        r_star: AbstractQuantity,
        v_inf: AbstractQuantity,
        v_phot: AbstractQuantity,
        nu_0: AbstractQuantity,
        mdot: AbstractQuantity,
        f_lu: float,
        lower_level_per_mass: AbstractQuantity,
        beta: float,
        stimulated: float = 1.0,
        n_l_exponent: float = 0.0,
    ) -> Self:
        """Build a scenario from physical wind and atomic-data quantities.

        The reference density follows from mass continuity (``mdot_scale = 1``) and
        the line strength from the Sobolev constant; see
        :func:`continuity_number_density` and :func:`sobolev_tau_scale`.

        Args:
            r_star: Stellar radius.
            v_inf: Terminal wind speed.
            v_phot: Photospheric velocity.
            nu_0: Line rest frequency.
            mdot: Mass-loss rate.
            f_lu: Absorption oscillator strength.
            lower_level_per_mass: Lower-level absorbers per unit mass (1 / mass).
            beta: Velocity-law exponent.
            stimulated: Stimulated-emission correction (default 1).
            n_l_exponent: Radial power-law tweak of the lower-level population.

        Returns:
            The corresponding scenario.
        """
        density_ref = continuity_number_density(mdot, r_star, v_inf, lower_level_per_mass)
        tau_scale = sobolev_tau_scale(f_lu, nu_0, density_ref, r_star, v_inf, stimulated=stimulated)
        return cls(
            r_star=r_star,
            v_inf=v_inf,
            v_phot=v_phot,
            nu_0=nu_0,
            density_ref=density_ref,
            beta=beta,
            mdot_scale=1.0,
            tau_scale=tau_scale,
            n_l_exponent=n_l_exponent,
        )


@dataclass(frozen=True)
class RunConfig:
    """Run settings shared by both solvers.

    Attributes:
        x_min: Lowest normalised frequency (units of ``v_inf``).
        x_max: Highest normalised frequency.
        n_bins: Number of frequency samples (bin centres).
        n_packets: Monte Carlo sample count per estimator (``method="mc"`` only).
        seed: PRNG seed (``method="mc"`` only).
    """

    x_min: float = -1.3
    x_max: float = 1.3
    n_bins: int = 101
    n_packets: int = 500_000
    seed: int = 0


class WindLineResult(NamedTuple):
    """Emergent line profile with a physical velocity axis and run metadata.

    Attributes:
        velocity: Doppler velocity axis (``x * v_inf``) as an ``unxt.Quantity``.
        flux: Continuum-normalised flux ``F / F_continuum`` (dimensionless), shape ``(n_bins,)``.
        scales: The reference scales used to non-dimensionalise the run.
        method: The solver that produced the profile.
    """

    velocity: AbstractQuantity
    flux: Array
    scales: ReferenceScales
    method: Method


def solve(
    scenario: WindLineScenario, config: RunConfig | None = None, method: Method = Method.REFERENCE
) -> WindLineResult:
    """Solve a wind-line scenario and return the emergent profile.

    Args:
        scenario: The physical wind + line description.
        config: Run settings; defaults to :class:`RunConfig` if omitted.
        method: ``Method.REFERENCE`` (deterministic) or ``Method.MC`` (Monte Carlo).

    Returns:
        The emergent profile with a physical velocity axis and run metadata.

    Raises:
        ValueError: If ``method`` is not a known solver.
    """
    config = config or RunConfig()
    scales = ReferenceScales.from_quantities(scenario.r_star, scenario.v_inf, scenario.nu_0, scenario.density_ref)
    wind = BetaLawWind(
        beta=jnp.asarray(scenario.beta),
        v_phot=scales.nondimensionalize(scenario.v_phot, Scale.VELOCITY),
        mdot_scale=jnp.asarray(scenario.mdot_scale),
    )
    line = LineData(tau_scale=jnp.asarray(scenario.tau_scale), n_l_exponent=jnp.asarray(scenario.n_l_exponent))

    centers = jnp.linspace(config.x_min, config.x_max, config.n_bins)
    if method is Method.REFERENCE:
        flux = solve_profile(wind, line, centers)
    elif method is Method.MC:
        half = 0.5 * (config.x_max - config.x_min) / (config.n_bins - 1)
        edges = jnp.concatenate([centers - half, centers[-1:] + half])
        band = (config.x_min - half, config.x_max + half)
        flux = run_profile(wind, line, edges, jax.random.key(config.seed), n_packets=config.n_packets, band=band)
    else:  # pragma: no cover - StrEnum guards the input
        raise ValueError(f"unknown method: {method}")

    velocity = scales.redimensionalize(centers, Scale.VELOCITY, unit="km / s")
    return WindLineResult(velocity=velocity, flux=flux, scales=scales, method=method)
