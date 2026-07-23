"""Wind dynamics: radiative force, and (later) the equation of motion.

The current medium (L2) is *prescribed*; this package is where the medium becomes
*solved* — the radiative acceleration is computed from the radiation field
(:func:`sobolev_line_force` today), fed to a pluggable equation of motion, and
iterated with the transfer solvers to a self-consistent :class:`~mcrtx.media.TabulatedWind`.
See ``docs/architecture.md`` (Medium dynamics) for the general design.
"""

from mcrtx.dynamics.force import sobolev_line_force

__all__ = ["sobolev_line_force"]
