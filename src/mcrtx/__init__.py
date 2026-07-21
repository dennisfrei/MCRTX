"""mcrtx: GPU-native, differentiable radiative transfer building blocks in JAX.

Layer model (see docs/concept.md):

- ``mcrtx.core``      L1: packet state, RNG discipline, estimators, scales
- ``mcrtx.media``     L2: media / geometry models (wind models, later grids)
- ``mcrtx.physics``   L3: solver-agnostic physics (opacities, source functions)
- ``mcrtx.solvers``   L4: transfer solvers (Monte Carlo, deterministic reference)
- ``mcrtx.scenarios`` L5: user-facing scenario presets and result objects
- ``mcrtx.units``     L5: unit boundary (unxt) and dimensionless reference scales
"""

from mcrtx._version import __version__

__all__ = ["__version__"]
