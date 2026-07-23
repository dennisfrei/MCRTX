"""Monte Carlo Sobolev packet propagator (MVP solver).

M0 realises the *single-scattering, core-illumination* model that the
:mod:`~mcrtx.solvers.reference` solver integrates deterministically — the two
agree within Monte Carlo noise (acceptance criterion, concept sec. 5).

Two estimators share one continuum normalisation:

- **Absorption** — genuine packet propagation. Continuum photons launch from the
  photosphere (``r = 1``) with a Lambertian flux weight, stream to their single
  forward resonance and transmit a smooth fraction ``e^{-tau}`` (D3); the
  surviving weight escapes at the launch frequency.
- **Emission** — the scattered light re-emitted by the source ``S(r) = W(r) I_core``.
  Resonance surfaces are sampled ``(p ∝ p, x)``, weighted by
  ``W(r) (1 - e^{-tau})`` and occulted behind the star (red core rays), then
  deposited at their frequency. This is a Monte Carlo estimate of the same
  emission integral the reference evaluates, so the two match by construction.

Direction/geometry sampling is parameter-free, so no gradient flows through it
(concept sec. 3.4). Full multiple scattering — the fixed-length ``lax.scan``
closing the ``j_bar`` <-> source-function loop — is the M1 extension.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from mcrtx.core import Estimators, PacketState, deposit_escape
from mcrtx.media import Medium
from mcrtx.physics import sobolev_tau
from mcrtx.physics.source import LineData
from mcrtx.solvers.reference import SourceModel, source_at

__all__ = ["run_profile"]


def _resonance_z(wind: Medium, p: Array, target: Array, z_lo: Array, z_far: float, n_bisect: int) -> Array:
    """Bisection for ``w(z) = target`` on ``[z_lo, z_far]`` (target >= 0).

    ``w(z) = v(r) z / r`` with ``r = sqrt(p^2 + z^2)`` is monotonic in ``z``.
    """

    def w_proj(z: Array) -> Array:
        r = jnp.sqrt(p**2 + z**2)
        return wind.v(r) * z / r

    def _step(_: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        overshoot = w_proj(mid) > target
        return jnp.where(overshoot, lo, mid), jnp.where(overshoot, mid, hi)

    lo, hi = jax.lax.fori_loop(0, n_bisect, _step, (z_lo, jnp.full_like(z_lo, z_far)))
    return 0.5 * (lo + hi)


def _sobolev_tau_at(wind: Medium, line: LineData, p: Array, z: Array) -> tuple[Array, Array]:
    """Sobolev optical depth at the resonance ``(p, z)`` and its radius (clamped >= 1)."""
    r = jnp.maximum(jnp.sqrt(p**2 + z**2), 1.0)
    mu = z / r
    n_l = wind.rho(r) * r**line.n_l_exponent
    tau = sobolev_tau(line.tau_scale, n_l, mu, wind.v(r) / r, wind.dv_dr(r))
    return tau, r


def run_profile(
    wind: Medium,
    line: LineData,
    x_edges: Array,
    key: Array,
    *,
    n_packets: int,
    band: tuple[float, float],
    p_max: float = 20.0,
    z_far: float = 1.0e3,
    n_bisect: int = 50,
    source: SourceModel = SourceModel.THIN,
    n_r: int = 256,
    n_mu: int = 64,
) -> Array:
    """Monte Carlo P-Cygni profile, normalised to the continuum.

    In a monotonic wind each photon Sobolev-scatters at most once, so the
    self-consistent M1 result differs from M0 only in the emission source
    function (``source``); the transmitted/absorption beam is unchanged.

    Args:
        wind: The medium (dimensionless); any `Medium` (analytic or tabulated).
        line: Resonance-line data (``tau_scale``, ``n_l_exponent``).
        x_edges: Monotonic frequency-bin edges in units of ``v_inf``, shape ``(Nx + 1,)``.
        key: Root PRNG key for the run.
        n_packets: Number of samples for each of the absorption and emission estimators.
        band: ``(x_lo, x_hi)`` sampling band; should cover ``x_edges``.
        p_max: Outer impact parameter for the emission integral.
        z_far: Line-of-sight half-extent bracketing the resonance search.
        n_bisect: Bisection iterations for the resonance location.
        source: Emission source closure (``THIN`` for M0, ``SELF_CONSISTENT`` for M1).
        n_r: Radius-grid points for the self-consistent source table.
        n_mu: Angular quadrature points for the self-consistent source.

    Returns:
        Normalised flux ``F(x) / F_continuum`` per bin, shape ``(Nx,)``.
    """
    k_launch, k_p, k_x = jax.random.split(key, 3)
    n_bins = x_edges.shape[0] - 1
    dummy = jnp.zeros((n_packets,))
    escaped = jnp.zeros((n_packets,), dtype=bool)  # alive == False => deposited

    def deposit(nu: Array, weight: Array) -> Array:
        est = Estimators.zeros(n_bins, 1)
        state = PacketState(r=dummy, mu=dummy, nu=nu, weight=weight, alive=escaped)
        return deposit_escape(est, state, x_edges).spectrum

    # --- Absorption: continuum photons transmitted through their forward resonance.
    packets = PacketState.launch_photospheric(k_launch, n_packets, band)
    mu0 = packets.mu  # cosine to radial at r = 1, in [0, 1]
    x = packets.nu
    flux_weight = mu0  # Lambertian flux sampling
    p = jnp.sqrt(jnp.maximum(1.0 - mu0**2, 0.0))
    z0 = mu0
    w_start = wind.v(jnp.asarray(1.0)) * mu0
    r_far = jnp.sqrt(p**2 + z_far**2)
    w_max = wind.v(r_far) * z_far / r_far
    has_res = (-x > w_start) & (-x < w_max)
    z_res = _resonance_z(wind, p, -x, z0, z_far, n_bisect)
    tau, _ = _sobolev_tau_at(wind, line, p, z_res)
    trans = jnp.exp(jnp.where(has_res, -tau, 0.0))
    flux_absorption = deposit(x, flux_weight * trans)
    continuum = deposit(x, flux_weight)

    # --- Emission: source S = W(r) I_core sampled over resonance surfaces (p ∝ p).
    pe = p_max * jnp.sqrt(jax.random.uniform(k_p, (n_packets,)))
    xe = jax.random.uniform(k_x, (n_packets,), minval=band[0], maxval=band[1])
    abs_xe = jnp.abs(xe)
    z_entry = jnp.sqrt(jnp.maximum(1.0 - pe**2, 0.0))
    r_entry = jnp.maximum(pe, 1.0)
    w_min = wind.v(r_entry) * z_entry / r_entry  # 0 for lobe rays (p >= 1)
    r_far_e = jnp.sqrt(pe**2 + z_far**2)
    w_max_e = wind.v(r_far_e) * z_far / r_far_e
    has_e = (abs_xe < w_max_e) & (abs_xe >= w_min)
    # Red frequencies (x > 0) on core rays (p < 1) resonate behind the star.
    visible = has_e & ~((pe < 1.0) & (xe > 0.0))
    ze = _resonance_z(wind, pe, abs_xe, z_entry, z_far, n_bisect)
    tau_e, r_e = _sobolev_tau_at(wind, line, pe, ze)
    s_e = source_at(wind, line, r_e, source, p_max, n_r, n_mu)
    emissivity = s_e * (1.0 - jnp.exp(jnp.where(has_e, -tau_e, 0.0)))
    weight_emit = 0.5 * p_max**2 * emissivity * visible
    flux_emission = deposit(xe, weight_emit)

    return (flux_absorption + flux_emission) / continuum
