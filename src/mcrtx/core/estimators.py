"""Output accumulators (estimators).

MVP: observer-frame frequency histogram of escaped packet weights
(the emergent line profile). Additionally the mean-intensity estimator
``j_bar`` per radial shell is *logged but unused* in M0 — it is the
prepared coupling point for the NLTE rate equations in M1.
"""

from __future__ import annotations

from typing import NamedTuple, Self

import jax.numpy as jnp
from jax import Array

from mcrtx.core.packets import PacketState

__all__ = ["Estimators", "deposit_escape", "deposit_resonance"]


class Estimators(NamedTuple):
    """Accumulated run outputs (dimensionless; converted at the API boundary).

    Attributes:
        spectrum: Escaped weight per observer-frame frequency bin, shape ``(n_nu,)``.
        j_bar: Line mean-intensity estimator per radial shell, shape
            ``(n_shells,)``. Logged in M0, consumed from M1 on.
    """

    spectrum: Array
    j_bar: Array

    @classmethod
    def zeros(cls, n_nu: int, n_shells: int) -> Self:
        """Create empty accumulators.

        Args:
            n_nu: Number of observer-frame frequency bins.
            n_shells: Number of radial shells for the ``j_bar`` estimator.

        Returns:
            Estimators with zero-filled arrays.
        """
        return cls(spectrum=jnp.zeros((n_nu,)), j_bar=jnp.zeros((n_shells,)))


def deposit_escape(est: Estimators, packets: PacketState, nu_edges: Array) -> Estimators:
    """Bin escaped packet weights into the observer-frame spectrum.

    Only packets with ``alive == False`` (escaped) and a frequency inside
    ``[nu_edges[0], nu_edges[-1])`` contribute; the rest add zero.

    Args:
        est: Accumulators to update.
        packets: Packet state carrying ``nu``, ``weight`` and ``alive``.
        nu_edges: Monotonic bin edges, shape ``(spectrum.size + 1,)``.

    Returns:
        Estimators with the updated ``spectrum``.
    """
    idx = jnp.searchsorted(nu_edges, packets.nu, side="right") - 1
    in_range = (packets.nu >= nu_edges[0]) & (packets.nu < nu_edges[-1])
    contrib = jnp.where((~packets.alive) & in_range, packets.weight, 0.0)
    idx = jnp.clip(idx, 0, est.spectrum.shape[0] - 1)
    return est._replace(spectrum=est.spectrum.at[idx].add(contrib))


def deposit_resonance(est: Estimators, shell_idx: Array, weight: Array) -> Estimators:
    """Accumulate resonance-passage weights into the per-shell ``j_bar`` estimator.

    Args:
        est: Accumulators to update.
        shell_idx: Integer shell index per contribution, shape ``(N,)``.
        weight: Weight deposited per contribution, shape ``(N,)``.

    Returns:
        Estimators with the updated ``j_bar``.
    """
    return est._replace(j_bar=est.j_bar.at[shell_idx].add(weight))
