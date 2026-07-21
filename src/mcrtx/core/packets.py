"""Photon packet state as a struct-of-arrays pytree.

Packets are never terminated (design decision D3): instead of discrete
absorption events they carry a continuous ``weight`` so that all gradients
flow through smooth expectation values. Escaped packets are frozen via
``alive`` masking inside a fixed-length ``lax.scan`` loop (D5).
"""

from __future__ import annotations

from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["PacketState"]


class PacketState(NamedTuple):
    """State of ``N`` photon packets; every field has shape ``(N,)``.

    All quantities are dimensionless w.r.t. the run's
    :class:`~mcrtx.units.ReferenceScales`.

    Attributes:
        r: Radius in units of the reference length.
        mu: Direction cosine w.r.t. the local radial direction, in ``[-1, 1]``.
        nu: Observer-frame frequency in units of the reference frequency.
        weight: Continuous packet weight (expectation-path gradients, D3).
        alive: Boolean mask; ``False`` once a packet has escaped or hit the star.
    """

    r: Array
    mu: Array
    nu: Array
    weight: Array
    alive: Array

    @classmethod
    def launch_photospheric(cls, key: Array, n_packets: int, band: tuple[float, float]) -> Self:
        """Launch ``n_packets`` from the photosphere ``r = 1``.

        Directions are isotropic into the outward hemisphere (``mu`` uniform on
        ``[0, 1]``, i.e. uniform in solid angle) and frequencies are flat over
        ``band`` around the line — the simple continuum start of concept sec. 7.2.
        Every packet starts with unit weight and alive.

        Args:
            key: PRNG key for direction and frequency sampling.
            n_packets: Number of packets to launch.
            band: ``(nu_lo, nu_hi)`` frequency band in units of the reference frequency.

        Returns:
            The initial packet state, every field of shape ``(n_packets,)``.
        """
        k_mu, k_nu = jax.random.split(key)
        mu = jax.random.uniform(k_mu, (n_packets,), minval=0.0, maxval=1.0)
        nu = jax.random.uniform(k_nu, (n_packets,), minval=band[0], maxval=band[1])
        return cls(
            r=jnp.ones((n_packets,)),
            mu=mu,
            nu=nu,
            weight=jnp.ones((n_packets,)),
            alive=jnp.ones((n_packets,), dtype=bool),
        )
