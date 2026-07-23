"""Multi-line spectrum synthesis (M2).

A spectrum usually shows several resonance lines, each at its own rest
frequency. On a shared velocity axis (units of ``v_inf``) line ``i`` sits at a
centre ``X_i`` — its Doppler offset from a reference wavelength,
``X_i = (lambda_i - lambda_ref)/lambda_ref * c/v_inf`` — and contributes its own
P-Cygni profile ``R_i(x - X_i)``.

Two ways to combine them:

- :func:`multiline_profile` **sums the deviations from the continuum**,
  ``R(x) = 1 + sum_i (R_i(x - X_i) - 1)`` — cheap (independent single-line
  solves) and *exact when the lines do not overlap*, but only first-order where
  they do (it can even drop below zero for overlapping saturated lines).
- :func:`multiline_profile_coupled` does the **exact sequential Sobolev formal
  solution** along each ray, so lines that share resonance zones (close doublets
  like C IV 1548/1550) are handled correctly. It reduces to the single-line
  solver for one line and to the additive result for well-separated lines.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from mcrtx.media import Medium
from mcrtx.physics import sobolev_tau
from mcrtx.physics.source import LineData
from mcrtx.solvers.reference import SourceModel, solve_profile, source_at

__all__ = ["multiline_profile", "multiline_profile_coupled"]


def multiline_profile(
    wind: Medium,
    centers: Sequence[float],
    lines: Sequence[LineData],
    x: Array,
    *,
    source: SourceModel = SourceModel.THIN,
    n_p: int = 800,
    n_bisect: int = 50,
) -> Array:
    """Composite normalised spectrum of several resonance lines.

    Args:
        wind: The beta-law wind medium (shared by all lines).
        centers: Line-centre velocities on the ``x`` axis, in units of ``v_inf``.
        lines: Per-line data; ``lines[i]`` pairs with ``centers[i]``.
        x: Output velocity axis (units of ``v_inf``), shape ``(Nx,)``.
        source: Source-function closure applied to every line.
        n_p: Impact-parameter samples for each line's reference solve.
        n_bisect: Bisection iterations for the resonance location.

    Returns:
        Normalised flux ``F(x) / F_continuum``, shape ``(Nx,)``.
    """
    flux = jnp.ones_like(x)
    for center, line in zip(centers, lines, strict=True):
        profile = solve_profile(wind, line, x - center, source=source, n_p=n_p, n_bisect=n_bisect)
        flux = flux + (profile - 1.0)
    return flux


def _trapz(y: Array, grid: Array) -> Array:
    dx = jnp.diff(grid)
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def _resonance_magnitude(wind: Medium, p: Array, target: Array, z_lo: Array, z_far: float, n_bisect: int) -> Array:
    """Bisection for ``|z|`` solving ``w(|z|) = target`` on ``[z_lo, z_far]``."""

    def w_proj(z: Array) -> Array:
        r = jnp.sqrt(p**2 + z**2)
        return wind.v(r) * z / r

    def _step(_: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        overshoot = w_proj(mid) > target
        return jnp.where(overshoot, lo, mid), jnp.where(overshoot, mid, hi)

    shape = jnp.broadcast_shapes(jnp.shape(p), jnp.shape(target), jnp.shape(z_lo))
    lo0 = jnp.broadcast_to(z_lo, shape)
    hi0 = jnp.broadcast_to(jnp.asarray(z_far), shape)
    lo, hi = jax.lax.fori_loop(0, n_bisect, _step, (lo0, hi0))
    return 0.5 * (lo + hi)


def multiline_profile_coupled(
    wind: Medium,
    centers: Sequence[float],
    lines: Sequence[LineData],
    x: Array,
    *,
    source: SourceModel = SourceModel.THIN,
    n_p: int = 800,
    p_max: float = 20.0,
    z_far: float = 1.0e3,
    n_bisect: int = 50,
    n_r: int = 256,
    n_mu: int = 64,
) -> Array:
    """Exact composite spectrum of overlapping resonance lines.

    Each ray meets one resonance per active line; the emergent intensity is the
    **sequential** Sobolev formal solution over them, which in closed form is

        I = I_bg * exp(-sum_i tau_i) + sum_i S_i (1 - e^{-tau_i}) exp(-sum_{z_j>z_i} tau_j),

    i.e. each line's emission is attenuated by the resonances closer to the
    observer. This handles lines that share resonance zones (close doublets)
    exactly; for well-separated lines it agrees with :func:`multiline_profile`
    but is more expensive (one coupled solve rather than independent ones).

    Args:
        wind: The beta-law wind medium (shared by all lines).
        centers: Line-centre velocities on the ``x`` axis, in units of ``v_inf``.
        lines: Per-line data; ``lines[i]`` pairs with ``centers[i]``.
        x: Output velocity axis (units of ``v_inf``), shape ``(Nx,)``.
        source: Source-function closure applied to every line.
        n_p: Number of impact-parameter samples.
        p_max: Outer impact parameter in units of the reference length.
        z_far: Line-of-sight half-extent bracketing the resonance search.
        n_bisect: Bisection iterations for the resonance location.
        n_r: Radius-grid points for a self-consistent/NLTE source table.
        n_mu: Angular quadrature points for a self-consistent/NLTE source.

    Returns:
        Normalised flux ``F(x) / F_continuum``, shape ``(Nx,)``.
    """
    p = jnp.linspace(0.0, p_max, n_p)  # (Np,)
    core = p < 1.0
    z_entry = jnp.sqrt(jnp.maximum(1.0 - p**2, 0.0))  # wind entry / front surface
    r_entry = jnp.maximum(p, 1.0)
    w_min = wind.v(r_entry) * z_entry / r_entry  # (Np,); 0 for lobe rays
    r_far = jnp.sqrt(p**2 + z_far**2)
    w_max = wind.v(r_far) * z_far / r_far
    pg = p[None, :]  # (1, Np)

    tau_stack, source_stack, z_stack = [], [], []
    for center, line in zip(centers, lines, strict=True):
        x_local = jnp.asarray(x)[:, None] - center  # (Nx, 1)
        abs_x = jnp.abs(x_local)
        has_res = (abs_x < w_max[None, :]) & (abs_x >= w_min[None, :])  # (Nx, Np)
        z_mag = _resonance_magnitude(wind, pg, abs_x, z_entry[None, :], z_far, n_bisect)
        z_signed = -jnp.sign(x_local) * z_mag  # front (z>0) if the line is blue here
        r_res = jnp.sqrt(pg**2 + z_mag**2)  # >= 1 by construction
        n_l = wind.rho(r_res) * r_res**line.n_l_exponent
        tau = sobolev_tau(line.tau_scale, n_l, z_mag / r_res, wind.v(r_res) / r_res, wind.dv_dr(r_res))
        # Core rays see only the front hemisphere; the back is occulted by the star.
        visible = jnp.where(core[None, :], (z_signed > 0.0) & has_res, has_res)
        tau_stack.append(jnp.where(visible, tau, 0.0))
        source_stack.append(source_at(wind, line, r_res, source, p_max, n_r, n_mu))
        z_stack.append(z_signed)

    tau = jnp.stack(tau_stack, axis=-1)  # (Nx, Np, n_lines)
    emit = jnp.stack(source_stack, axis=-1) * (1.0 - jnp.exp(-tau))
    z_res = jnp.stack(z_stack, axis=-1)

    # Emission of line i is attenuated by every resonance closer to the observer.
    closer = z_res[..., None, :] > z_res[..., :, None]  # (Nx, Np, n_i, n_j): z_j > z_i
    atten_emit = jnp.exp(-jnp.sum(tau[..., None, :] * closer, axis=-1))  # (Nx, Np, n_i)
    background = jnp.where(core, 1.0, 0.0)[None, :] * jnp.exp(-jnp.sum(tau, axis=-1))
    intensity = background + jnp.sum(emit * atten_emit, axis=-1)  # (Nx, Np)

    flux = _trapz(intensity * pg, p)
    flux_continuum = _trapz(jnp.where(core, 1.0, 0.0) * p, p)
    return flux / flux_continuum
