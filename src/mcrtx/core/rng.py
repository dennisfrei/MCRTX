"""PRNG key discipline (concept sec. 3.5).

One root key per run. Every random draw derives its key deterministically
from ``(root, packet_id, interaction_step, stream)`` so results are fully
reproducible and independent of execution order, and common random numbers
across parameter sets are possible for variance-reduced gradient checks.
"""

from __future__ import annotations

import jax

__all__ = ["interaction_key"]


def interaction_key(root: jax.Array, step: int, stream: int) -> jax.Array:
    """Derive the key for one interaction step and logical stream.

    Per-packet independence is obtained by the caller via
    ``jax.random.split(key, n_packets)`` on the returned key, keeping the
    derivation order-independent.

    Args:
        root: Root PRNG key of the run.
        step: Interaction step index within the fixed-length propagation loop.
        stream: Logical stream id (e.g. 0 = scatter decision, 1 = re-emission angle).

    Returns:
        The folded PRNG key for ``(step, stream)``.
    """
    return jax.random.fold_in(jax.random.fold_in(root, step), stream)
