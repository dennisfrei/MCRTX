"""Deterministic SEI-style reference solver (validation baseline, D6).

Sobolev line formation in p-z ray geometry following Lamers, Cerruti-Sola &
Perinotto (1987), sharing the L2/L3 modules with the MC solver — the first
proof of the multi-method design. Clarity over speed.

Geometry: the observer looks along ``+z``; a ray has impact parameter ``p`` and
radius ``r = sqrt(p^2 + z^2)``. The projected outflow velocity toward the
observer is ``w(z) = v(r) * z / r``; it is odd in ``z`` and (for a monotonic
wind) monotonic, so a photon at normalised frequency ``x`` (Doppler units of
``v_inf``) has a single resonance where ``w = -x``. Blue frequencies
(``x < 0``) resonate in the front hemisphere (``z > 0``); red frequencies in
the occulted-for-core back hemisphere. Each ray carries one Sobolev
transmission/emission event ``I -> I e^{-tau} + S (1 - e^{-tau})`` with
``S(r) = W(r) I_core``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from mcrtx.media import BetaLawWind
from mcrtx.physics import sobolev_tau
from mcrtx.physics.source import LineData

__all__ = ["solve_profile"]

_I_CORE = 1.0  # continuum core intensity; sets the flux normalisation.


def _trapz(y: Array, grid: Array) -> Array:
    """Trapezoidal integral of ``y`` over the last axis against 1-D ``grid``."""
    dx = jnp.diff(grid)
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def solve_profile(
    wind: BetaLawWind,
    line: LineData,
    x: Array,
    *,
    n_p: int = 800,
    p_max: float = 20.0,
    z_far: float = 1.0e3,
    n_bisect: int = 50,
) -> Array:
    """Emergent P-Cygni line profile, normalised to the continuum.

    Args:
        wind: The beta-law wind medium (dimensionless).
        line: Resonance-line data (``tau_scale``, ``n_l_exponent``).
        x: Normalised observer-frame frequencies in units of ``v_inf``, shape ``(Nx,)``.
        n_p: Number of impact-parameter samples.
        p_max: Outer impact parameter in units of the reference length.
        z_far: Line-of-sight half-extent used to bracket the resonance search.
        n_bisect: Bisection iterations for the resonance location.

    Returns:
        Normalised flux ``F(x) / F_continuum``, shape ``(Nx,)``. Unity means the
        continuum; below unity is absorption, above unity is emission.
    """
    p = jnp.linspace(0.0, p_max, n_p)
    core = p < 1.0
    pg = p[None, :]  # (1, Np)
    xg = jnp.asarray(x)[:, None]  # (Nx, 1)
    absx = jnp.abs(xg)
    blue = xg < 0.0

    # Wind entry point along each ray: r = 1 for core rays (p < 1), r = p otherwise.
    z_entry = jnp.sqrt(jnp.maximum(1.0 - pg**2, 0.0))  # (1, Np)
    r_entry = jnp.maximum(pg, 1.0)

    def w_proj(z: Array) -> Array:
        r = jnp.sqrt(pg**2 + z**2)
        return wind.v(r) * z / r

    # A resonance |w| = |x| is reachable between the wind entry and the far field.
    w_min = wind.v(r_entry) * z_entry / r_entry  # (1, Np); 0 for lobe rays
    r_far = jnp.sqrt(pg**2 + z_far**2)
    w_max = wind.v(r_far) * z_far / r_far
    has_res = (absx < w_max) & (absx >= w_min)

    # Bisect for the front-hemisphere resonance z >= z_entry with w(z) = |x|.
    ones = jnp.ones_like(xg)

    def _step(_: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        overshoot = w_proj(mid) > absx
        return jnp.where(overshoot, lo, mid), jnp.where(overshoot, mid, hi)

    lo0 = ones * z_entry
    hi0 = ones * jnp.full_like(pg, z_far)
    lo, hi = jax.lax.fori_loop(0, n_bisect, _step, (lo0, hi0))
    z_res = 0.5 * (lo + hi)
    r_res = jnp.sqrt(pg**2 + z_res**2)  # >= 1 by construction

    mu = z_res / r_res
    n_l = wind.rho(r_res) * r_res**line.n_l_exponent
    tau = sobolev_tau(line.tau_scale, n_l, mu, wind.v(r_res) / r_res, wind.dv_dr(r_res))
    tau = jnp.where(has_res, tau, 0.0)
    trans = jnp.exp(-tau)
    emit = wind.dilution(r_res) * _I_CORE * (1.0 - trans)

    # Core rays carry the continuum; a front (blue) resonance attenuates it.
    # Red frequencies on core rays resonate behind the star and are occulted.
    core_intensity = jnp.where(blue & has_res, _I_CORE * trans + emit, _I_CORE)
    # Lobe rays have no background; both hemispheres emit.
    lobe_intensity = jnp.where(has_res, emit, 0.0)
    intensity = jnp.where(core[None, :], core_intensity, lobe_intensity)

    flux = _trapz(intensity * pg, p)
    flux_continuum = _trapz(jnp.where(core, _I_CORE, 0.0) * p, p)
    return flux / flux_continuum
