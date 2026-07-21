"""Photon packet state as a struct-of-arrays pytree.

Packets are never terminated (design decision D3): instead of discrete
absorption events they carry a continuous ``weight`` so that all gradients
flow through smooth expectation values. Escaped packets are frozen via
``alive`` masking inside a fixed-length ``lax.scan`` loop (D5).
"""

from __future__ import annotations

from typing import NamedTuple

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

    # TODO(M0): factory ``launch_photospheric(key, n_packets, band)`` sampling
    #   flat-in-frequency around nu_0, outward isotropic (concept sec. 7.2).
