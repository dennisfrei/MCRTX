"""L3 solver-agnostic physics modules (design decision D2).

Modules here answer local questions only — "what is tau_sob / the source
function at this position for this direction" — and must not know whether
a Monte Carlo packet or a deterministic ray is asking.
"""

from mcrtx.physics.nlte import AtomicModel, assemble_rate_matrix, statistical_equilibrium
from mcrtx.physics.sobolev import sobolev_tau
from mcrtx.physics.source import (
    LineData,
    sobolev_escape_probabilities,
    sobolev_nlte_source,
    two_level_source,
)

__all__ = [
    "AtomicModel",
    "LineData",
    "assemble_rate_matrix",
    "sobolev_escape_probabilities",
    "sobolev_nlte_source",
    "sobolev_tau",
    "statistical_equilibrium",
    "two_level_source",
]
