"""MVP scenario: single resonance line in a 1D stellar wind.

The only place where units, defaults and solver selection meet. Returns a
result object carrying the emergent profile as an ``unxt.Quantity`` plus
run metadata (seed, scales, config) for full reproducibility.
"""

from __future__ import annotations

__all__: list[str] = []

# TODO(M0): ``WindLineScenario`` dataclass (unxt quantities in, validated),
#   ``solve(scenario, config, method="mc" | "reference")``.
