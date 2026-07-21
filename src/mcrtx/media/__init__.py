"""L2 media / geometry models.

A medium exposes pure, differentiable functions of position — for the 1D
wind MVP: ``v(r)``, ``dv_dr(r)``, ``rho(r)``, ``dilution(r)``. Solvers and
physics modules consume only this interface, never model internals.
"""

from mcrtx.media.wind import BetaLawWind

__all__ = ["BetaLawWind"]
