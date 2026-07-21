"""L3 solver-agnostic physics modules (design decision D2).

Modules here answer local questions only — "what is tau_sob / the source
function at this position for this direction" — and must not know whether
a Monte Carlo packet or a deterministic ray is asking.
"""

from mcrtx.physics.sobolev import sobolev_tau
from mcrtx.physics.source import LineData

__all__ = ["LineData", "sobolev_tau"]
