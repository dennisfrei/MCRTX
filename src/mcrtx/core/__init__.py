"""L1 compute core: packet state, RNG discipline, estimators.

Nothing in this package knows about specific physics or media models.
All structures are JAX pytrees with static shapes (design decision D5).
"""

from mcrtx.core.packets import PacketState
from mcrtx.core.rng import interaction_key

__all__ = ["PacketState", "interaction_key"]
