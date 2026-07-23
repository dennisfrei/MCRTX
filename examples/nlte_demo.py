"""NLTE playground: multi-level statistical equilibrium, LTE vs radiation-pumped.

The solver is domain-agnostic — atom + up/down rate coefficients in, populations
out. Here a 3-level toy atom sits in collisional (LTE) equilibrium, then a
growing radiation field pumps the 0->1 transition out of LTE.

    uv run --group examples python examples/nlte_demo.py   # + PNG
    uv run python examples/nlte_demo.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, sparkline

from mcrtx.physics.nlte import AtomicModel, assemble_rate_matrix, statistical_equilibrium

# --- CONSTANTS: edit me ------------------------------------------------------
G = [1.0, 3.0, 5.0]  # statistical weights
ENERGY = [0.0, 1.0, 2.5]  # level energies (scaled by kT unit below)
KT = 0.7  # temperature (same energy units)
C_DOWN = [1.0, 2.0, 0.5]  # collisional down-rate per transition (0-1, 1-2, 0-2)
A_UL = 1.0  # spontaneous emission on the pumped 0-1 line
PUMP = [0.0, 0.3, 1.0, 3.0, 10.0]  # radiation-field strengths J to sweep
# -----------------------------------------------------------------------------

LOWER = [0, 1, 0]
UPPER = [1, 2, 2]


def main() -> None:
    g = jnp.asarray(G)
    energy = jnp.asarray(ENERGY)
    lower = jnp.asarray(LOWER)
    upper = jnp.asarray(UPPER)
    model = AtomicModel(g=g, lower=lower, upper=upper)

    # Detailed-balance collisional rates -> LTE (Boltzmann) on their own.
    boltz_factor = (g[upper] / g[lower]) * jnp.exp(-(energy[upper] - energy[lower]) / KT)
    c_down = jnp.asarray(C_DOWN)
    c_up = c_down * boltz_factor

    boltzmann = g * jnp.exp(-energy / KT)
    boltzmann = boltzmann / jnp.sum(boltzmann)
    console.print(f"LTE (Boltzmann) populations: {[round(float(v), 3) for v in boltzmann]}", markup=False)
    console.print("radiation pumps the 0->1 line; populations leave LTE as J grows:\n", markup=False)

    def solve(pump: float) -> jax.Array:
        # Radiative rates on transition 0 (0->1): r_up = (g_u/g_l) J, r_down = A + J.
        r_up = c_up + jnp.asarray([(g[1] / g[0]) * pump, 0.0, 0.0])
        r_down = c_down + jnp.asarray([A_UL + pump, 0.0, 0.0])
        return statistical_equilibrium(assemble_rate_matrix(model, r_up, r_down))

    populations = [solve(p) for p in PUMP]
    for pump, n in zip(PUMP, populations, strict=True):
        console.print(f"  J={pump:5.1f}: n = [{', '.join(f'{float(v):.3f}' for v in n)}]", markup=False)

    n_upper = [float(n[1]) for n in populations]
    console.print(f"\n  n(level 1) vs J: {sparkline(n_upper)}", markup=False)

    png = save_png(
        "examples/output/nlte.png",
        PUMP,
        {f"n(level {i})": [float(n[i]) for n in populations] for i in range(len(G))},
        xlabel=r"Radiation field $J$ (scaled)",
        ylabel=r"Level population $n_i$",
        title=r"3-level atom: LTE $\to$ radiation-pumped",
    )
    console.print(f"\nPNG: {png}" if png else "\n(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
