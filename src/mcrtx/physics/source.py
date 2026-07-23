"""Source functions / level populations (exchangeable module, D2).

Two-level atom, escape-probability closure (Sobolev 1960; Castor 1970):

- M0 :func:`~mcrtx.media.wind.BetaLawWind.dilution` gives the optically-thin
  source ``S = W(r) I_core`` (first scattering only).
- M1 :func:`two_level_source` solves the self-consistent pure-scattering source
  ``S = (beta_c / beta) I_core``, which reduces to ``W(r)`` when thin.

The interface is deliberately not pinned to two levels: an NLTE rate-equation
solver (Sobolev-beta coupled, one batched linear system per shell) replaces this
module later without touching the core API.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from mcrtx.physics.nlte import AtomicModel, assemble_rate_matrix, statistical_equilibrium

__all__ = [
    "LineData",
    "escape_probability",
    "sobolev_escape_probabilities",
    "sobolev_nlte_source",
    "two_level_source",
]


class LineData(NamedTuple):
    """Static data of a single resonance line (dimensionless).

    The trailing fields parameterise the two-level NLTE source
    (:func:`sobolev_nlte_source`); their defaults give pure scattering, so the
    ``THIN``/``SELF_CONSISTENT`` source models ignore them.

    Attributes:
        tau_scale: Line-strength scale entering
            :func:`mcrtx.physics.sobolev.sobolev_tau`.
        n_l_exponent: Population model: ``n_l(r) ∝ rho(r)`` scaled by this power-law tweak.
        collision: Collisional de-excitation rate ``C_ul / A_ul`` (0 = pure scattering).
        w_core: Core radiation occupation number ``I_core / (2 h nu^3 / c^2)``.
        boltzmann: Detailed-balance factor ``exp(-h nu / k T_gas)`` for the collisions.
        g_lower: Lower-level statistical weight.
        g_upper: Upper-level statistical weight.
    """

    tau_scale: Array
    n_l_exponent: Array
    collision: Array | float = 0.0
    w_core: Array | float = 1.0
    boltzmann: Array | float = 1.0
    g_lower: float = 1.0
    g_upper: float = 3.0


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


def sobolev_escape_probabilities(
    tau_scale: Array,
    n_l: Array,
    v_over_r: Array | float,
    dv_dr: Array | float,
    mu_star: Array,
    *,
    n_mu: int = 64,
) -> tuple[Array, Array]:
    """Angle-averaged Sobolev escape probability ``beta`` and its core-cone part ``beta_c``.

    Both are quadratures of :func:`escape_probability` over the direction cosine;
    ``tau_mu`` is even in ``mu``, so ``beta = int_0^1 beta(tau_mu) dmu`` and
    ``beta_c = (1/2) int_{mu_star}^1 beta(tau_mu) dmu``. Evaluate one point at a
    time (``vmap`` over radii).

    Args:
        tau_scale: Dimensionless line-strength scale.
        n_l: Lower-level population at the point (reference-density units).
        v_over_r: Local ``v / r``.
        dv_dr: Local radial velocity gradient.
        mu_star: Core-edge cosine ``sqrt(1 - 1/r^2)``.
        n_mu: Number of angular quadrature points.

    Returns:
        The pair ``(beta, beta_c)``.
    """
    t = jnp.linspace(0.0, 1.0, n_mu)
    dt = 1.0 / (n_mu - 1)

    def _beta(mu: Array) -> Array:
        grad = jnp.abs(mu**2 * dv_dr + (1.0 - mu**2) * v_over_r)
        return escape_probability(tau_scale * n_l / grad)

    def _trapz01(y: Array) -> Array:
        return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]), axis=-1) * dt

    beta = _trapz01(_beta(t))  # (1/2) int_{-1}^{1} = int_0^1
    mu_c = mu_star + (1.0 - mu_star) * t  # remap the core cone [mu_star, 1] to [0, 1]
    beta_c = 0.5 * (1.0 - mu_star) * _trapz01(_beta(mu_c))
    return beta, beta_c


def two_level_source(
    tau_scale: Array,
    n_l: Array,
    v_over_r: Array | float,
    dv_dr: Array | float,
    mu_star: Array,
    *,
    n_mu: int = 64,
) -> Array:
    """Self-consistent two-level Sobolev source function ``S / I_core`` (pure scattering).

    ``S = J_bar`` with ``J_bar = (1 - beta) S + beta_c I_core`` gives
    ``S = (beta_c / beta) I_core``, reducing to the geometric dilution ``W(r)`` in
    the optically-thin limit. This is :func:`sobolev_nlte_source` with no collisions.

    Args:
        tau_scale: Dimensionless line-strength scale.
        n_l: Lower-level population at the point (reference-density units).
        v_over_r: Local ``v / r``.
        dv_dr: Local radial velocity gradient.
        mu_star: Core-edge cosine ``sqrt(1 - 1/r^2)``.
        n_mu: Number of angular quadrature points.

    Returns:
        The source function in units of the core intensity ``I_core``.
    """
    beta, beta_c = sobolev_escape_probabilities(tau_scale, n_l, v_over_r, dv_dr, mu_star, n_mu=n_mu)
    return beta_c / beta


def sobolev_nlte_source(
    tau_scale: Array,
    n_l: Array,
    v_over_r: Array | float,
    dv_dr: Array | float,
    mu_star: Array,
    *,
    collision: Array | float = 0.0,
    w_core: Array | float = 1.0,
    boltzmann: Array | float = 1.0,
    g_lower: float = 1.0,
    g_upper: float = 3.0,
    n_mu: int = 64,
) -> Array:
    """Two-level Sobolev line source ``S / I_core`` from the NLTE rate equations.

    Couples the wind to :func:`~mcrtx.physics.nlte.statistical_equilibrium`: the
    Sobolev escape probabilities set the effective radiative rates (net escape
    ``beta A_ul`` and core pumping ``beta_c``), collisions add a thermal channel,
    and the resulting level populations fix the source function. With
    ``collision = 0`` it reduces to :func:`two_level_source`; as collisions
    dominate the populations thermalise (LTE) toward the Boltzmann ratio.

    All rates are scaled by the spontaneous rate ``A_ul``.

    Args:
        tau_scale: Dimensionless line-strength scale.
        n_l: Lower-level population at the point (reference-density units).
        v_over_r: Local ``v / r``.
        dv_dr: Local radial velocity gradient.
        mu_star: Core-edge cosine ``sqrt(1 - 1/r^2)``.
        collision: Collisional de-excitation rate ``C_ul / A_ul`` (0 = pure scattering).
        w_core: Core radiation occupation number ``I_core / (2 h nu^3 / c^2)``.
        boltzmann: Detailed-balance factor ``exp(-h nu / k T_gas)`` for the collisions.
        g_lower: Lower-level statistical weight.
        g_upper: Upper-level statistical weight.
        n_mu: Number of angular quadrature points.

    Returns:
        The source function in units of the core intensity ``I_core``.
    """
    beta, beta_c = sobolev_escape_probabilities(tau_scale, n_l, v_over_r, dv_dr, mu_star, n_mu=n_mu)
    g_ratio = g_upper / g_lower
    # Effective scaled rates: radiative (escape + core pumping) + collisional.
    r_up = beta_c * g_ratio * w_core + collision * g_ratio * boltzmann
    r_down = beta + beta_c * w_core + collision
    model = AtomicModel(g=jnp.asarray([g_lower, g_upper]), lower=jnp.asarray([0]), upper=jnp.asarray([1]))
    populations = statistical_equilibrium(assemble_rate_matrix(model, jnp.atleast_1d(r_up), jnp.atleast_1d(r_down)))
    excitation = g_ratio * populations[0] / populations[1]  # (g_u n_l) / (g_l n_u)
    return (1.0 / w_core) / (excitation - 1.0)
