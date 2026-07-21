"""L1 compute core: packet state, RNG discipline, estimators.

Nothing in this package knows about specific physics or media models.
All structures are JAX pytrees with static shapes (design decision D5).
"""

from mcrtx.core.estimators import Estimators, deposit_escape, deposit_resonance
from mcrtx.core.packets import PacketState
from mcrtx.core.rng import interaction_key

__all__ = [
    "Estimators",
    "PacketState",
    "deposit_escape",
    "deposit_resonance",
    "interaction_key",
]
